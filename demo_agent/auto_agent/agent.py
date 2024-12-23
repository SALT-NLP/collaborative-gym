import argparse
import json
import os
import re
from typing import Dict, Union, List

import dspy
import toml
import yaml
from aact.cli.launch.launch import _sync_run_node
from aact.cli.reader import NodeConfig
from aact.cli.reader.dataflow_reader import NodeArgs
from knowledge_storm.lm import VLLMClient, OpenAIModel, ClaudeModel, TogetherClient

from collaborative_gym.utils.context_processing import ContextProcessor
from demo_agent.utils.memory import Scratchpad


class ReactAutoAgent:
    """A fully autonomous agent implemented with ReAct prompting strategy.
    This agent only interacts with the environment to complete the task and does not consider collaborating with human.

    The agent is enhanced with a scratchpad. The agent decides whether to update the scratchpad after each action.
    """

    def __init__(self,
                 lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
                 add_task_demo,
                 prompt_path='demo_agent/auto_agent/prompts.yaml'):
        self.name = None
        self.task_description = None
        self.task_action_space_description = None
        self.lm = lm
        self.scratchpad = Scratchpad()
        self.action_history = []

        self.add_task_demo = add_task_demo
        self.task_demo = None

        with open(prompt_path, 'r') as f:
            prompts = yaml.safe_load(f)

        self.act_prompt_template = "\n\n".join([
            prompts["system_template"],
            prompts["action_history_template"],
            prompts["take_next_task_action_template"],
        ])
        self.update_scratchpad_prompt_template = "\n\n".join([
            prompts["system_template"],
            prompts["update_scratchpad_template"],
        ])

        self.context_processor = ContextProcessor()

    def format_act_prompt(self, obs: Dict, action_space_description: str):
        return self.act_prompt_template.format(
            name=self.name,
            task_description=self.task_description,
            action_space_description=action_space_description,
            scratchpad=self.scratchpad.to_str(),
            observation=self.context_processor.observation_to_str(obs),
            action_history=self.context_processor.action_history_to_str(self.action_history),
        )

    def format_update_scratchpad_prompt(self, obs: Dict):
        return self.update_scratchpad_prompt_template.format(
            name=self.name,
            task_description=self.task_description,
            scratchpad=self.scratchpad.to_str(),
            observation=self.context_processor.observation_to_str(obs),
            scratchpad_action_space_description=self.scratchpad.get_action_space_description()
        )

    def start(self,
              name: str,
              team_members: List[str],
              task_description: str,
              action_space: dict,
              example_question: str,
              example_trajectory: List):
        """Start the agent with the given task information.

        This function will be called by collaborative_gym.nodes_agent_interface when the agent is started.
        """
        self.name = name
        # This is a fully autonomous agent, so it does not use `team_members`.
        self.task_description = task_description
        self.task_action_space_description = self.context_processor.action_space_to_str(
            action_space=action_space,
            excluded_action_names=[]
        )
        self.task_demo = self.context_processor.task_example_to_str(
            example_question=example_question,
            example_trajectory=example_trajectory
        )
        if self.add_task_demo:
            self.act_prompt_template = "\n\n".join([
                self.act_prompt_template,
                self.task_demo,
                "Now give your output starting with \"Thought:\"."
            ])

        print('Agent started.')  # TODO: better logging

    def get_action(self, observation: Dict, chat_history: list):
        """Get the next action from the agent.

        This function will be called by collaborative_gym.nodes_agent_interface when the node receives a new observation from
        the environment.
        """
        # Update the scratchpad
        scratchpad_update_prompt = self.format_update_scratchpad_prompt(obs=observation)
        scratchpad_update_prompt_response = self.lm(
            prompt=scratchpad_update_prompt,
            temperature=0,
            max_tokens=4000
        )
        scratchpad_update = scratchpad_update_prompt_response[0].strip()
        scratchpad_update = scratchpad_update[scratchpad_update.find('Action:') + len('Action:'):].strip()
        self.scratchpad.execute_action(scratchpad_update)

        # Take the next action
        if len(chat_history) > 0:
            observation["environment_message"] = [c["message"] for c in chat_history]
        act_prompt = self.format_act_prompt(
            obs=observation,
            action_space_description=self.task_action_space_description,
        )
        act_prompt_response = self.lm(
            prompt=act_prompt,
            temperature=0,
            max_tokens=4000
        )
        action = act_prompt_response[0].strip()
        action = action[action.find('Action:') + len('Action:'):].strip()
        # Hacky post-processing:
        # Assume the action is in a function call format and the function name starts with a capital letter.
        if "\nThought:" in action:
            action = action[:action.find("\nThought:")].strip()
        match = re.search(r'[A-Z]', action)
        if match:
            action = action[match.start():]
        if action[-1] != ')':
            action = action[:action.rfind(')') + 1]
        action = action.replace("\(", "(").replace("\)", ")")

        # Claude tend to generate code that leads to syntax error in jupyter notebook
        action = action.replace("print(\"\n", "print(\"").replace("print(\"\\n", "print(\"")

        print(f'Agent action: {action}')
        self.action_history.append(action)

        return action

    def end(self, result_dir: str):
        os.makedirs(os.path.join(result_dir, self.name), exist_ok=True)
        with open(os.path.join(result_dir, self.name, 'info.json'), 'w') as f:
            info = {
                "lm": self.lm.kwargs["model"],
                "token_usage": self.get_token_usage(),
            }
            json.dump(info, f, indent=4)
        with open(os.path.join(result_dir, self.name, 'scratchpad.json'), 'w') as f:
            json.dump(self.scratchpad.notes, f, indent=4)
        with open(os.path.join(result_dir, self.name, 'llm_call_history.jsonl'), 'w') as f:
            for call in self.lm.history:
                f.write(json.dumps(call) + '\n')

    def get_token_usage(self):
        return {
            'prompt_tokens': self.lm.prompt_tokens,
            'completion_tokens': self.lm.completion_tokens
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='gpt-4o-2024-08-06')
    parser.add_argument('--prompt-path', type=str, default='demo_agent/auto_agent/prompts.yaml')
    parser.add_argument('--use-vllm', action='store_true', default=False)
    parser.add_argument('--use-together', action='store_true', default=False)
    parser.add_argument('--lm-url', type=str, default='http://localhost',
                        help='URL to the language model server.')
    parser.add_argument('--lm-port', type=int, default=8000,
                        help='Port to the language model server.')
    parser.add_argument('--wait-time', type=int, default=5,
                        help='Time to wait for the agent to respond. This is useful when coordinating with human.')
    parser.add_argument('--node-name', type=str, required=True)
    parser.add_argument('--env-uuid', type=str, required=True)
    parser.add_argument('--redis-url', type=str, default='redis://localhost:6379/0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    secrets = toml.load('secrets.toml')
    for k in secrets:
        os.environ[k] = secrets[k]

    if args.use_vllm:
        lm = VLLMClient(
            model=args.model_name,
            url=args.lm_url,
            port=args.lm_port
        )
    elif args.use_together:
        lm = TogetherClient(
            model=args.model_name,
            api_key=os.environ['TOGETHER_API_KEY']
        )
        lm.kwargs["model"] = args.model_name
    elif 'gpt' in args.model_name:
        lm = OpenAIModel(
            model=args.model_name,
            api_key=os.environ['OPENAI_API_KEY'],
        )
    elif 'claude' in args.model_name:
        lm = ClaudeModel(
            model=args.model_name,
            api_key=os.environ['ANTHROPIC_API_KEY']
        )
    else:
        raise ValueError(f'Unsupported model name: {args.model_name}')

    if args.debug:
        agent = ReactAutoAgent(
            lm=lm,
            add_task_demo=True,
            prompt_path=args.prompt_path
        )

        import pdb

        pdb.set_trace()
    else:
        _sync_run_node(
            NodeConfig(
                node_name=args.node_name,
                node_class="agent",
                node_args=NodeArgs(
                    env_uuid=args.env_uuid,
                    node_name=args.node_name,
                    agent=ReactAutoAgent(
                        lm=lm,
                        add_task_demo=True,
                        prompt_path=args.prompt_path
                    ),
                    wait_time=args.wait_time
                )
            ),
            args.redis_url
        )
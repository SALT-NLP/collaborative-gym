import argparse
import json
import logging
import os
import re
from typing import Dict, List

import knowledge_storm.lm
import toml
import yaml
from aact.cli.launch.launch import _sync_run_node
from aact.cli.reader import NodeConfig
from aact.cli.reader.dataflow_reader import NodeArgs
from knowledge_storm.lm import LitellmModel

from collaborative_gym.core import (
    SendTeammateMessage,
    WaitTeammateContinue,
    RequestTeammateConfirm,
)
from collaborative_gym.utils.context_processing import ContextProcessor
from collaborative_gym.utils.utils import prepare_lm_kwargs
from demo_agent.utils.memory import Scratchpad

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)


class CollaborativeAgent:
    """A basic collaborative agent that first plan to (1) take collaborative action, or (2) take task action, or
    (3) do nothing, before generating the actual action string.

    The agent is enhanced with a scratchpad. The agent decides whether to update the scratchpad after each action.
    """

    def __init__(
        self,
        lm: knowledge_storm.lm.LM,
        add_plan_next_step_demo,
        add_task_demo,
        enhance_user_control,
        prompt_path="demo_agent/collaborative_agent_with_situational_planning/prompts.yaml",
    ):
        self.name = None
        self.team_members = None
        self.task_description = None
        self.task_action_space_description = None
        self.lm = lm
        self.scratchpad = Scratchpad()
        self.action_history = []
        self.collaboration_acts = {
            "send_teammate_message": SendTeammateMessage(),
            "wait_teammate_continue": WaitTeammateContinue(),
            "request_teammate_confirm": RequestTeammateConfirm(),
        }
        self.add_plan_next_step_demo = add_plan_next_step_demo
        self.add_task_demo = add_task_demo
        self.task_demo = None
        self.enhance_user_control = enhance_user_control

        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)

        if self.add_plan_next_step_demo:
            self.plan_prompt_template = "\n\n".join(
                [
                    prompts["system_template"],
                    prompts["action_history_template"],
                    prompts["plan_next_step_template"],
                    prompts["plan_next_step_demonstration"],
                    'Now give your output starting with "Thought:".',
                ]
            )
        else:
            self.plan_prompt_template = "\n\n".join(
                [
                    prompts["system_template"],
                    prompts["action_history_template"],
                    prompts["plan_next_step_template"],
                    'Now give your output starting with "Thought:".',
                ]
            )
        self.act_prompt_template = "\n\n".join(
            [
                prompts["system_template"],
                prompts["action_history_template"],
                prompts["take_next_task_action_template"],
            ]
        )
        self.message_prompt_template = "\n\n".join(
            [
                prompts["system_template"],
                prompts["send_message_template"],
            ]
        )
        self.update_scratchpad_prompt_template = "\n\n".join(
            [
                prompts["system_template"],
                prompts["update_scratchpad_template"],
            ]
        )

        self.context_processor = ContextProcessor()

    def format_plan_prompt(self, obs: Dict, chat_history: list):
        return self.plan_prompt_template.format(
            name=self.name,
            team_members=", ".join(self.team_members),
            task_description=self.task_description,
            scratchpad=self.scratchpad.to_str(),
            chat_history=self.context_processor.chat_history_to_str(
                self.name, chat_history
            ),
            observation=self.context_processor.observation_to_str(obs),
            action_history=self.context_processor.action_history_to_str(
                self.action_history
            ),
        )

    def format_act_prompt(
        self, obs: Dict, chat_history: list, action_space_description: str
    ):
        return self.act_prompt_template.format(
            name=self.name,
            team_members=", ".join(self.team_members),
            task_description=self.task_description,
            action_space_description=action_space_description,
            scratchpad=self.scratchpad.to_str(),
            chat_history=self.context_processor.chat_history_to_str(
                self.name, chat_history
            ),
            observation=self.context_processor.observation_to_str(obs),
            action_history=self.context_processor.action_history_to_str(
                self.action_history
            ),
        )

    def format_message_prompt(self, obs: Dict, chat_history: list):
        return self.message_prompt_template.format(
            name=self.name,
            team_members=", ".join(self.team_members),
            task_description=self.task_description,
            scratchpad=self.scratchpad.to_str(),
            chat_history=self.context_processor.chat_history_to_str(
                self.name, chat_history
            ),
            observation=self.context_processor.observation_to_str(obs),
        )

    def format_update_scratchpad_prompt(self, obs: Dict, chat_history: list):
        return self.update_scratchpad_prompt_template.format(
            name=self.name,
            team_members=", ".join(self.team_members),
            task_description=self.task_description,
            scratchpad=self.scratchpad.to_str(),
            chat_history=self.context_processor.chat_history_to_str(
                self.name, chat_history
            ),
            observation=self.context_processor.observation_to_str(obs),
            scratchpad_action_space_description=self.scratchpad.get_action_space_description(),
        )

    def start(
        self,
        name: str,
        team_members: List[str],
        task_description: str,
        action_space: dict,
        example_question: str,
        example_trajectory: List,
    ):
        """Start the agent with the given task information.

        This function will be called by collaborative_gym.nodes_agent_interface when the agent is started.
        """
        self.name = name
        self.team_members = team_members
        self.task_description = task_description
        self.task_action_space_description = self.context_processor.action_space_to_str(
            action_space=action_space,
            excluded_action_names=[
                self.collaboration_acts[k].human_readable_name
                for k in self.collaboration_acts
            ],
        )
        self.task_demo = self.context_processor.task_example_to_str(
            example_question=example_question, example_trajectory=example_trajectory
        )
        if self.add_task_demo:
            self.act_prompt_template = "\n\n".join(
                [
                    self.act_prompt_template,
                    self.task_demo,
                    'Now give your output starting with "Thought:".',
                ]
            )

        logger.info("Collaborative Agent with Situational Planning started.")

    def get_action(self, observation: Dict, chat_history: list):
        """Get the next action from the agent.

        This function will be called by collaborative_gym.nodes_agent_interface when the node receives a new observation from
        the environment.
        """
        try:
            # Update the scratchpad
            scratchpad_update_prompt = self.format_update_scratchpad_prompt(
                obs=observation, chat_history=chat_history
            )
            try:
                scratchpad_update_prompt_response = self.lm(
                    prompt=scratchpad_update_prompt, temperature=0, max_tokens=4000
                )
                scratchpad_update = scratchpad_update_prompt_response[0].strip()
                if "Action:" in scratchpad_update:
                    scratchpad_update = scratchpad_update[
                        scratchpad_update.find("Action:") + len("Action:") :
                    ].strip()
                self.scratchpad.execute_action(scratchpad_update)
            except Exception as e:
                logger.error(f"Error updating scratchpad: {e}")

            # Take the next action
            try:
                plan_prompt = self.format_plan_prompt(
                    obs=observation, chat_history=chat_history
                )
                plan_prompt_response = self.lm(
                    prompt=plan_prompt, temperature=0, max_tokens=1000
                )
                plan = plan_prompt_response[0].strip()
            except Exception as e:
                logger.error(f"Error generating plan: {e}")
                plan = "2"  # Default to taking task action
            if "Plan:" in plan:
                plan = plan[plan.find("Plan:") + len("Plan:") :].strip()
            else:
                plan = "2"
            if "1" in plan:
                message_prompt = self.format_message_prompt(
                    obs=observation, chat_history=chat_history
                )
                message_prompt_response = self.lm(
                    prompt=message_prompt, temperature=0, max_tokens=4000
                )
                message = message_prompt_response[0].strip()
                if "Message:" in message:
                    message = message[message.find("Message:") + len("Message:") :].strip()
                action = self.collaboration_acts[
                    "send_teammate_message"
                ].construct_action_string_from_params(message=message)
            elif "2" in plan:
                act_prompt = self.format_act_prompt(
                    obs=observation,
                    action_space_description=self.task_action_space_description,
                    chat_history=chat_history,
                )
                act_prompt_response = self.lm(
                    prompt=act_prompt, temperature=0, max_tokens=4000
                )
                action = act_prompt_response[0].strip()
                if "Action:" in action:
                    action = action[action.find("Action:") + len("Action:") :].strip()
                # Hacky post-processing:
                # Assume the action is in a function call format and the function name starts with a capital letter.
                if "\nThought:" in action:
                    action = action[: action.find("\nThought:")].strip()
                match = re.search(r"[A-Z]", action)
                if match:
                    action = action[match.start() :]
                if action[-1] != ")":
                    action = action[: action.rfind(")") + 1]
                action = action.replace("\(", "(").replace("\)", ")")

                # Claude tend to generate code that leads to syntax error in jupyter notebook
                # Handle notebook code transformations
                action = action.replace('print("\n', 'print("').replace(
                    'print("\\n', 'print("'
                )
                if self.enhance_user_control:
                    if action == "FINISH()":
                        action = self.collaboration_acts[
                            "wait_teammate_continue"
                        ].construct_action_string_from_params()  # Don't allow the agent to finish
                    if "EDITOR_UPDATE" in action:
                        action = self.collaboration_acts[
                            "request_teammate_confirm"
                        ].construct_action_string_from_params(
                            request_id="editor_update", pending_action=action
                        )
            else:
                action = self.collaboration_acts[
                    "wait_teammate_continue"
                ].construct_action_string_from_params()
        except Exception as e:
            logger.error(f"Error generating action: {e}")
            # action = self.collaboration_acts[
            #     "wait_teammate_continue"
            # ].construct_action_string_from_params()
            action = self.collaboration_acts[
                "send_teammate_message"
            ].construct_action_string_from_params(
                message="I'm sorry, I encountered an error while processing. Could you take an action yourself?"
            )

        logger.info(f"Collaborative Agent with Situational Planning action: {action}")
        self.action_history.append(action)

        return action

    def end(self, result_dir: str):
        os.makedirs(os.path.join(result_dir, self.name), exist_ok=True)
        with open(os.path.join(result_dir, self.name, "info.json"), "w") as f:
            info = {
                "lm": self.lm.model,
                "token_usage": self.get_token_usage(),
            }
            json.dump(info, f, indent=4)
        with open(os.path.join(result_dir, self.name, "scratchpad.json"), "w") as f:
            json.dump(self.scratchpad.notes, f, indent=4)
        with open(
            os.path.join(result_dir, self.name, "llm_call_history.jsonl"), "w"
        ) as f:
            for call in self.lm.history:
                f.write(
                    json.dumps({"prompt": call["prompt"], "response": call["response"]})
                    + "\n"
                )
        logger.info("Collaborative Agent with Situational Planning ended.")

    def get_token_usage(self):
        return {
            "prompt_tokens": self.lm.prompt_tokens,
            "completion_tokens": self.lm.completion_tokens,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4o-2024-08-06",
        help="We use LiteLLM to dispatch the request to the correct model."
        "Please ensure the model name matches the naming convention in LiteLLM."
        "https://docs.litellm.ai/docs/providers",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="demo_agent/collaborative_agent_with_situational_planning/prompts.yaml",
    )
    parser.add_argument(
        "--wait-time",
        type=int,
        default=5,
        help="Time to wait for the agent to respond. This is useful when coordinating with human.",
    )
    parser.add_argument("--node-name", type=str, required=True)
    parser.add_argument("--env-uuid", type=str, required=True)
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--add-task-demo", action="store_true")
    parser.add_argument(
        "--enhance-user-control",
        action="store_true",
        help="If set, the agent cannot choose to finish the task proactively and will always ask for confirmation "
        "before updating the editor.",
    )
    args = parser.parse_args()

    secrets = toml.load("secrets.toml")
    for k in secrets:
        os.environ[k] = secrets[k]

    lm_kwargs = prepare_lm_kwargs(args.model_name)
    lm = LitellmModel(**lm_kwargs)

    if args.debug:
        agent = CollaborativeAgent(
            lm=lm,
            add_plan_next_step_demo=True,
            add_task_demo=args.add_task_demo,
            enhance_user_control=args.enhance_user_control,
            prompt_path=args.prompt_path,
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
                    agent=CollaborativeAgent(
                        lm=lm,
                        add_plan_next_step_demo=True,
                        add_task_demo=args.add_task_demo,
                        enhance_user_control=args.enhance_user_control,
                        prompt_path=args.prompt_path,
                    ),
                    wait_time=args.wait_time,
                ),
            ),
            args.redis_url,
        )

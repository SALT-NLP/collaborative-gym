import json
import os
import random
import re
import shutil
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from knowledge_storm import OpenAIModel

from collaborative_gym.core import CoEnv, ObservationTypes, logger
from collaborative_gym.envs.registry import EnvFactory
from collaborative_gym.spaces import (
    MAX_UNICODE_LENGTH,
    MultiSpace,
    UnicodeWithRegexPattern,
)
from collaborative_gym.utils.code_executor import JupyterManager
from collaborative_gym.utils.file_system import clear_directory
from collaborative_gym.utils.string import post_process_parsed_function_arg
from collaborative_gym.utils.text_editor import TextEditor

DISCOVERY_BENCH_DATAPOINTS = {
    "introduction_pathways_non-native_plants": {
        "metadata_0.json": 3,
        "metadata_1.json": 2,
        "metadata_2.json": 2,
        "metadata_3.json": 3,
        "metadata_4.json": 3,
        "metadata_5.json": 3,
    },
    "archaeology": {
        "metadata_0.json": 1,
        "metadata_1.json": 1,
        "metadata_2.json": 1,
        "metadata_3.json": 1,
        "metadata_4.json": 1,
        "metadata_5.json": 1,
        "metadata_6.json": 1,
        "metadata_7.json": 1,
        "metadata_8.json": 1,
        "metadata_9.json": 1,
        "metadata_10.json": 1,
        "metadata_11.json": 1,
        "metadata_12.json": 1,
        "metadata_13.json": 1,
        "metadata_14.json": 1,
        "metadata_15.json": 1,
        "metadata_16.json": 1,
        "metadata_17.json": 1,
        "metadata_18.json": 1,
        "metadata_19.json": 1,
        "metadata_20.json": 1,
        "metadata_21.json": 1,
        "metadata_22.json": 1,
        "metadata_23.json": 1,
        "metadata_24.json": 1,
        "metadata_25.json": 1,
        "metadata_26.json": 1,
        "metadata_27.json": 1,
        "metadata_28.json": 1,
        "metadata_29.json": 1,
        "metadata_30.json": 1,
        "metadata_31.json": 1,
        "metadata_32.json": 1,
        "metadata_33.json": 1,
        "metadata_34.json": 1,
        "metadata_35.json": 1,
        "metadata_36.json": 1,
        "metadata_37.json": 1,
    },
    "meta_regression_raw": {
        "metadata_0.json": 4,
        "metadata_1.json": 2,
        "metadata_2.json": 2,
        "metadata_3.json": 3,
        "metadata_4.json": 1,
        "metadata_5.json": 3,
        "metadata_6.json": 3,
        "metadata_7.json": 2,
        "metadata_8.json": 2,
        "metadata_9.json": 1,
        "metadata_10.json": 2,
        "metadata_11.json": 2,
        "metadata_12.json": 2,
        "metadata_13.json": 4,
        "metadata_14.json": 1,
        "metadata_15.json": 3,
        "metadata_16.json": 2,
        "metadata_17.json": 3,
        "metadata_18.json": 3,
        "metadata_19.json": 5,
    },
    "worldbank_education_gdp_indicators": {
        "metadata_0.json": 2,
        "metadata_1.json": 1,
        "metadata_2.json": 1,
        "metadata_3.json": 1,
        "metadata_4.json": 1,
    },
}

DISCOVERY_BENCH_DATAPOINTS_LIST = [
    (k, task_id, i)
    for k, v in DISCOVERY_BENCH_DATAPOINTS.items()
    for task_id, query_cnt in v.items()
    for i in range(query_cnt)
]


class CoAnalysisActions(Enum):
    EXECUTE_JUPYTER_CELL = "EXECUTE_JUPYTER_CELL"
    EDITOR_UPDATE = "EDITOR_UPDATE"
    FINISH = "FINISH"

    def __str__(self):
        return self.value


@EnvFactory.register("tabular_analysis")
class CoAnalysisEnv(CoEnv):
    """
    ## Description
    CoAnalysisEnv is a collaborative environment for analyzing tabular data.
    The environment supports executing code in Jupyter notebook and documenting findings.

    ## Action Space
    Actions are strings that must match one of the following patterns:
    - EXECUTE_JUPYTER_CELL(code: str): Run Python code in a Jupyter cell
    - EDITOR_UPDATE(text: str): Update the editor content
    - FINISH(): Complete the current task

    ## Observation Space
    The observation is a dictionary containing:
    - jupyter_history (non-private): History of executed code cells and their outputs
    - result_editor (non-private): Current state of the analysis document
    """

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        use_simulated_dataset: bool = False,
        csv_files: Optional[List[str]] = None,
        query: Optional[str] = None,
        discovery_bench_data_point_idx: Optional[int] = None,  # [0, 1, ..., 109]
        discovery_bench_root_dir: str = "datasets/discoverybench/real/test",
        docker_image_name: str = "cogym-jupyter-cpu-image",
        docker_local_root_dir: str = os.path.join(os.getcwd(), "tmp"),
    ):
        super().__init__(team_members=team_members, env_id=env_id)

        self.use_simulated_dataset = use_simulated_dataset
        self.dataset_root_dir = discovery_bench_root_dir

        if self.use_simulated_dataset:
            self.discovery_bench_data_point = DISCOVERY_BENCH_DATAPOINTS_LIST[
                discovery_bench_data_point_idx
            ]
            with open(
                os.path.join(
                    discovery_bench_root_dir,
                    self.discovery_bench_data_point[0],
                    self.discovery_bench_data_point[1],
                )
            ) as f:
                self.discovery_bench_metadata = json.load(
                    f
                )  # metadata in DiscoveryBench
            self.query = self.discovery_bench_metadata["queries"][0][
                self.discovery_bench_data_point[2]
            ]["question"]
            ground_truth_df = pd.read_csv(
                os.path.join(self.dataset_root_dir, "answer_key_real.csv")
            )
            task_id = int(
                self.discovery_bench_data_point[1].split(".")[0].split("_")[1]
            )
            self.discovery_bench_gold_hypo = ground_truth_df.loc[
                (ground_truth_df["dataset"] == self.discovery_bench_data_point[0])
                & (ground_truth_df["metadataid"] == task_id)
                & (ground_truth_df["query_id"] == self.discovery_bench_data_point[2]),
                "gold_hypo",
            ].values[0]

            try:
                # The evaluator could be changed to a different model.
                # Use OpenAI GPT to match the Collaborative Gym paper.
                self.evaluator_lm = OpenAIModel(
                    model="gpt-4o-2024-08-06",
                    api_key=os.environ["OPENAI_API_KEY"],
                )
            except KeyError:
                self.evaluator_lm = None
                logger.error(
                    "Please provide your OpenAI API key in the environment variable OPENAI_API_KEY to enable the evaluator."
                )

        else:
            self.query = query

        # Docker Jupyter sandbox for executing Python code
        self.docker_volume_local_dir = os.path.join(docker_local_root_dir, self.env_id)
        os.makedirs(self.docker_volume_local_dir, exist_ok=True)
        self.jupyter_manager = JupyterManager(
            custom_image_name=docker_image_name,
            docker_volume_local_dir=self.docker_volume_local_dir,
            timeout=60 * 30,  # set a long timeout
        )
        self.docker_volume_container_dir = self.jupyter_manager.docker_server.volumes[
            self.docker_volume_local_dir
        ]["bind"]

        # Task information
        self.dataset_local_paths = []
        self.datasets = []
        if self.use_simulated_dataset:
            for d in self.discovery_bench_metadata["datasets"]:
                self.dataset_local_paths.append(
                    os.path.join(
                        self.dataset_root_dir,
                        self.discovery_bench_data_point[0],
                        d["name"],
                    )
                )
                self.datasets.append(
                    os.path.join(self.docker_volume_container_dir, d["name"])
                )

            self.additional_task_info = {
                "domain_knowledge": self.discovery_bench_metadata.get(
                    "domain_knowledge", ""
                ),
                "full_datasets_description": self.discovery_bench_metadata["datasets"],
            }
        else:
            for f in csv_files:
                self.dataset_local_paths.append(os.path.join(self.dataset_root_dir, f))
                self.datasets.append(
                    os.path.join(self.docker_volume_container_dir, f.split("/")[-1])
                )

        self.task_description = (
            "Your task is to analyze the provided tabular dataset(s):\n"
            f"{self.datasets}\n"
        )
        if self.use_simulated_dataset:
            # Additional task description to specify the anaysis result format that can be parsed by DiscoveryBench evaluation script
            self.task_description += (
                "Specifically, you need to discover hypothesis to answer the following query:\n"
                f"{self.query}\n"
                "In the final answer, please write down a scientific hypothesis in natural "
                "language, derived from the provided dataset, clearly stating the context of "
                "hypothesis (if any), variables chosen (if any) and relationship between those "
                "variables (if any) including any statistical significance.\n"
                "Here is an example for what the final answer shall look like:\n"
                "Query: What is the linear coefficient that describes the positive relationship "
                "between the rate of maximum body length evolution and spatial variation in "
                "speciation rates, where the former emerges as the most influential factor?\n"
                "Final Answer: The linear coefficient that describes the positive relationship "
                "between the rate of maximum body length evolution ('BAMM_speciation') and "
                "spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.\n"
                "Please make sure to provide a clear and concise answer to the query following "
                "the example.\n"
            )

        else:
            self.task_description += f"Specifically, here is the user query that you need to follow:\n{self.query}"
        self.task_description += (
            "\n\nMake sure the final result is included in the result editor. When you are done, "
            "your performance will be evaluated based on the content in the result editor."
        )

        # An example question and trajectory for team members to understand the task
        self.example_question = (
            "Load all datasets using python using provided paths. Paths: /Users/bodhi/projects"
            "/datavoyager/DiscoveryBench/evolution_freshwater_fish/"
            "body-size-evolution-in-south-american-freshwater-fishes.csv. What is the linear "
            "coefficient that describes the positive relationship between the rate of maximum "
            "body length evolution and spatial variation in speciation rates, where the "
            "former emerges as the most influential factor?"
        )
        self.example_trajectory = [
            (
                "First, I need to load the dataset from the provided path using Python, and then analyze the data "
                "to find the linear coefficient that describes the relationship between the rate of maximum body "
                "length evolution and spatial variation in speciation rates.",
                "EXECUTE_JUPYTER_CELL(code=\"import pandas as pd\n\ndf = pd.read_csv('/Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()\")",
                {
                    "jupyter_history": """"Code block:\nimport pandas as pd\n\ndf = pd.read_csv('/Users/bodhi/projects/datavoyager/DiscoveryBench/evolution_freshwater_fish/body-size-evolution-in-south-american-freshwater-fishes.csv')\ndf.head()\nOutput:
            HYBAS_ID       long       lat  BAMM_speciation  BAMM_extinction  BAMM_NetDiv  ...   aet    Elevation  sgr  soil_div     area  diversity
        0  6050000010 -76.477422  7.742693         0.137392         0.026807     0.110585  ...  1387   330.150088  166  0.482402  72363.7         68
        1  6050000740 -74.628725  9.803586         0.117235         0.025796     0.091438  ...  1082    69.475294   23  0.457436  17944.3         35
        2  6050068100 -75.295995  8.448815         0.119381         0.023826     0.095555  ...  1312   143.032178   74  0.378793  17105.5         44
        3  6050068110 -74.608408  8.922863         0.132477         0.027777     0.104700  ...  1445    14.724138    3  0.468328    610.1         48
        4  6050070260 -75.591588  5.770093         0.120127         0.022940     0.097187  ...  1371  1378.729945  421  0.158870  61901.9         81
        [5 rows x 21 columns]"""
                },
            ),
            (
                "Now that the dataset is loaded, I need to perform a linear regression analysis to determine the coefficient that describes the relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv').",
                "EXECUTE_JUPYTER_CELL(code=\"from sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\ncoefficient\")",
                {
                    "jupyter_history": """Code block:\nfrom sklearn.linear_model import LinearRegression\nimport numpy as np\n\n# Prepare the data for regression\nX = df[['BAMM_speciation']].values.reshape(-1, 1)  # Predictor\ny = df['BAMM_NetDiv'].values  # Response\n\n# Create a linear regression model\nmodel = LinearRegression()\nmodel.fit(X, y)\n\n# Get the coefficient\ncoefficient = model.coef_[0]\ncoefficient\nOutput:\n0.5175306498596297"""
                },
            ),
            (
                "Now, I have enough results to come up with the hypothesis to answer the query. I will write down the hypothesis in the result editor.",
                "EDITOR_UPDATE(text=\"The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518.\")",
                {
                    "result_editor": "The linear coefficient that describes the positive relationship between the rate of maximum body length evolution ('BAMM_speciation') and spatial variation in speciation rates ('BAMM_NetDiv') is approximately 0.518."
                },
            ),
        ]

        self.result_editor = TextEditor()

        # Action space
        self.action_space = MultiSpace(
            (
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(
                        r"^EXECUTE_JUPYTER_CELL\(code=(.*)\)$", re.DOTALL
                    ),
                    params=["code"],
                    machine_readable_identifier=CoAnalysisActions.EXECUTE_JUPYTER_CELL,
                    human_readable_name="Execute code in Jupyter cell",
                    human_readable_description="Execute the given Python code or Jupyter magic command in a Jupyter "
                    "cell based on the current Jupyter execution context. Make sure the "
                    "code satisfy the syntax restrictions of the Jupyter notebook. "
                    "The Jupyter executor stores the execution history so that you can "
                    "use the results of previous executions in the current cell.\n"
                    f"Do not print long outputs as it will exceed the character limit. For "
                    f"file saving, you can only access {self.docker_volume_container_dir}.",
                ),
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(
                        r"^EDITOR_UPDATE\(text=(.*)\)$", re.DOTALL
                    ),
                    params=["text"],
                    machine_readable_identifier=CoAnalysisActions.EDITOR_UPDATE,
                    human_readable_name="Update the result editor",
                    human_readable_description="Update the result editor with the provided text. The full original"
                    " text will be replaced.",
                ),
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(r"^FINISH\(\)$", re.DOTALL),
                    params=[],
                    machine_readable_identifier=CoAnalysisActions.FINISH,
                    human_readable_name="Finish the task",
                    human_readable_description="Finish the data analysis task.",
                ),
            )
        )

        self.private_action_space = MultiSpace(())  # No private actions

    # Shared actions
    def _execute_jupyter_cell(self, code: str):
        self.jupyter_manager.execute_python_code(code)

    def _editor_update(self, text: str):
        self.result_editor.update_text(text)

    def close(self):
        self.jupyter_manager.close()

    def get_obs(self):
        obs = {
            "public": {
                "jupyter_history": self.jupyter_manager.execution_history_to_str(),
                "result_editor": self.result_editor.get_text(),
            },
            "private": {team_member: {} for team_member in self.team_members},
        }
        return obs

    def obs_type(self) -> dict[str, ObservationTypes]:
        return {
            "jupyter_history": ObservationTypes.JUPYTER_NOTEBOOK,
            "result_editor": ObservationTypes.TEXT_EDITOR,
            "domain_knowledge": ObservationTypes.NO_RENDER,
            "full_datasets_description": ObservationTypes.NO_RENDER,
        }

    def reset(
        self,
        options: dict[str, Any] | None = None,
    ):
        clear_directory(self.docker_volume_local_dir)
        # Mount the data
        for local_path, container_path in zip(self.dataset_local_paths, self.datasets):
            # Docker volume local dir will be mounted to the container dir.
            shutil.copy(
                local_path,
                container_path.replace(
                    self.docker_volume_container_dir, self.docker_volume_local_dir
                ),
            )
        # Initialize the Jupyter cells
        self.jupyter_manager.reset()
        obs = self.get_obs()

        return obs, {}

    def step(self, role: str, action: str):
        """Execute one timestep within the environment.

        Args:
            role (str): The team member executing the action
            action (str): The action to take, formatted as a string matching one of the action space patterns

        Returns:
            observation, reward, termination state, private, additional information
        """
        info = {}
        info["action_start_time"] = time.time()

        # Parse and validate action using parent class helper
        parsed_action, private, action_id, err_msg = self.parse_and_validate_action(
            role, action
        )
        if err_msg:
            return self.handle_action_error(err_msg, private)

        # Post-process parsed action parameters
        for k in parsed_action:
            parsed_action[k] = post_process_parsed_function_arg(parsed_action[k])

        info["action"] = action_id

        # Execute the action
        terminated = False
        reward = 0
        info["action_error"] = None
        try:
            if info["action"] == CoAnalysisActions.EXECUTE_JUPYTER_CELL:
                self._execute_jupyter_cell(code=parsed_action["code"])
            elif info["action"] == CoAnalysisActions.EDITOR_UPDATE:
                self._editor_update(text=parsed_action["text"])
            elif info["action"] == CoAnalysisActions.FINISH:
                terminated = True
        except Exception as e:
            err_msg = f"Error in executing the action: {action}. Error: {e}"
            return self.handle_action_error(err_msg, private)
        finally:
            info["action_end_time"] = time.time()

        # Get the observation
        obs = self.get_obs()

        return obs, reward, terminated, private, info

    @staticmethod
    def eval_helper_prepare_dataset_metadata_json(
        dataset_meta, dataset_type, use_column_metadata=True
    ):
        """Copied from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
        `prepare_dataset_metadata_json`."""
        if dataset_meta == None:
            return [
                {
                    "dataset_description": "",
                    "columns": [],
                }
            ]
        datasets_json = []
        if dataset_type == "real":
            for d in dataset_meta["datasets"]:
                datasets_json.append(
                    {
                        "dataset_description": d["description"],
                        "columns": (
                            [
                                {"name": col["name"], "description": col["description"]}
                                for col in d["columns"]["raw"]
                            ]
                            if use_column_metadata
                            else []
                        ),
                    }
                )
        else:
            for d in dataset_meta["datasets"]:
                datasets_json.append(
                    {
                        "dataset_description": d["description"],
                        "columns": (
                            [
                                {"name": col["name"], "description": col["description"]}
                                for col in d["columns"]
                            ]
                            if use_column_metadata
                            else []
                        ),
                    }
                )
        return datasets_json

    def eval_helper_get_response_with_retry(
        self,
        prompt,
        decode_output_to_json,
        temperature=None,
        max_retry=5,
        verbose=False,
    ):
        """Adapted from https://github.com/allenai/discoverybench/blob/main/utils/openai_helpers.py"""
        n_try = 0
        openai_gen_hyp = {
            "temperature": 0 if temperature is None else temperature,  # Set temperature as 0 to increase stability.
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        while n_try < max_retry:
            openai_gen_hyp["temperature"] += +0.0001 * random.uniform(
                0, 1
            )  # Walk-around for dspy cache
            try:
                output = self.evaluator_lm(prompt=prompt, **openai_gen_hyp)[0].strip()
            except Exception as e:
                if verbose:
                    print(f"Error in getting response: {e}")
                n_try += 1
                if n_try < max_retry:
                    if verbose:
                        print("Retrying...")
                else:
                    if verbose:
                        print("Retry limit reached")
                continue

            if decode_output_to_json:
                output = output.strip("```json").strip("```").strip()
                try:
                    response_json = json.loads(output)
                    return response_json
                except ValueError:
                    if verbose:
                        print(f"Bad JSON output:\n\n{output}")
                    n_try += 1
                    if n_try < max_retry:
                        if verbose:
                            print("Retrying...")
                    else:
                        if verbose:
                            print("Retry limit reached")
            else:
                return output

        return None

    def eval_helper_get_sub_hypotheses(
        self, hypo, dataset_meta, workflow="", use_column_metadata=True, max_retry=1
    ):
        """Get the sub-hypotheses from the hypothesis and the workflow.

        This function is adapted from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
            `get_sub_hypotheses`.
        """
        extraction_prompt = f"""\
            Given a set of dataset columns, a ground-truth hypothesis, and the analysis workflow used, your task is to extract three dimensions that define the hypothesis: Context, Variables, and Relations. \
            Here are the definitions for these dimensions:
            - Contexts: Boundary conditions that limit the scope of a hypothesis. E.g., “for men over \
            the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then extract the context from the dataset_descrption.
            - Variables: Known concepts that interact in a meaningful way under a given context to \
            produce the hypothesis. E.g., gender, age, income, or "None" if there is no interacting variable.
            - Relations: Interactions between a given set of variables under a given context to produce \
            the hypothesis. E.g., “quadratic relationship”, “inversely proportional”, piecewise conditionals, \
            or "None" if there is no interacting relationship.
            Make sure to only use the information present in the hypothesis and the workflow. Do not add any new information. \
            For each dimension, be specific, and do not omit any important details.

            Here is the metadata for the task:
            ```json
            {{
            "datasets": %s,
            "hypothesis": "%s",
            "workflow": "%s"
            }}
            ```

            Return your answer as a JSON object in the following format:
            ```json
            {{
            "sub_hypo": [
                {{
                    "text": the hypothesis in natural language,
                    "context": a short text description of the context of the hypothesis,
                    "variables": a list of columns involved in the hypothesis,
                    "relations": a short text description of the relationship between the variables of the hypothesis
                }},
                ...
            ]
            }}```
            """
        datasets_json = self.eval_helper_prepare_dataset_metadata_json(
            dataset_meta, dataset_type="real", use_column_metadata=use_column_metadata
        )
        _prompt = extraction_prompt % (datasets_json, hypo, workflow)
        # sub_hypo_json = get_response(client, _prompt, model=llm_used, max_retry=1)

        openai_gen_hyp = {  # Follow https://github.com/allenai/discoverybench/blob/main/utils/openai_helpers.py
            "temperature": 0,  # Set temperature as 0 to increase stability.
            "max_tokens": 4096,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }

        sub_hypo_json = self.eval_helper_get_response_with_retry(
            _prompt, decode_output_to_json=True, max_retry=max_retry
        )
        if sub_hypo_json is not None:
            # print(f"full hypothesis: {hypo}")
            print(f"sub_hypo_json: {sub_hypo_json}")
        else:
            sub_hypo_json = {
                "sub_hypo": [],
            }

        sub_hypo_json["full_hypo"] = hypo

        return sub_hypo_json

    def eval_helper_is_matching_context(
        self, gold_hyp, gold_context, pred_hyp, pred_context
    ):
        """This function is adapted from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
        `is_matching_context`."""
        if gold_context == pred_context:
            return True
        if "None" in [gold_context, pred_context]:
            return False
        prompt = f"""\
                Given a gold hypothesis, a gold context, a predicted hypothesis, and a predicted context, your task is \
                to determine if the predicted context semantically matches the ground-truth context. \
                Here is the definition for Context: Boundary conditions that limit the scope of a sub-hypothesis. E.g., “for men over the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then the context is derived from the dataset_descrption. \
                Here is the definition for Context: Boundary conditions that limit the scope of a sub-hypothesis. E.g., “for men over the age of 30”, “in Asia and Europe”. If the context applies to the full dataset, then the context is derived from the dataset_descrption. \
                If the predicted context matches the gold context, return true, otherwise return false.
                If both gold and predicted hypotheses are defined over the context of the full dataset, then also return true.
                If both gold and predicted hypotheses are defined over the context of the full dataset, then also return true.

                Here is the metadata for the task:
                ```json
                {{
                    "gold_hypothesis": "{gold_hyp}",
                    "gold_context": "{gold_context}",
                    "predicted_hypothesis": "{pred_hyp}",
                    "predicted_context": "{pred_context}"
                }}
                ```

                Return your answer as a JSON object in the following format:
                ```json
                {{
                    "match": true or false
                }}
                ```"""

        output = self.eval_helper_get_response_with_retry(
            prompt, decode_output_to_json=True, max_retry=5
        )
        if output:
            return output.get("match", False)
        return False

    @staticmethod
    def eval_helper_get_score_from_answer(type, answer):
        """Copied from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py `get_score_from_answer`."""
        if type == "context":
            answer = answer.replace("Answer:", "").strip()
            if answer.startswith("A)"):
                return 1.0
            elif answer.startswith("B)"):
                return 0.0
            return -1.0

        elif type == "var":
            try:
                answer = answer.strip("```json").strip("```").strip()
                var_json = json.loads(answer)
                p = 0.0
                r = 0.0
                f1 = 0.0
                if var_json["sizeB"]:
                    p = var_json["intersection"] / var_json["sizeB"]
                if var_json["sizeA"]:
                    r = var_json["intersection"] / var_json["sizeA"]
                if p > 0.0 and r > 0.0:
                    f1 = (2 * p * r) / (p + r)
                else:
                    f1 = 0.0
                eval_rec = {
                    "p": p,
                    "r": r,
                    "f1": f1,
                    "sizeA": var_json["sizeA"],
                    "sizeB": var_json["sizeB"],
                    "intersection": var_json["intersection"],
                    "explanation": var_json["explanation"],
                }
                return eval_rec
            except:
                return {"p": -1.0, "r": -1.0, "f1": -1.0}
        elif type == "rel":
            answer = answer.strip("```json").strip("```").strip()
            rel_json = json.loads(answer)
            answer_str = rel_json["answer"].strip()
            if answer_str.startswith("A") or "very similar" in answer_str:
                return 1.0
            elif (
                answer_str.startswith("B")
                or "similar but general than HypoA" in answer_str
            ):
                return 0.5
            elif answer_str.startswith("C") or "different" in answer_str:
                return 0.0
            return -1.0
        return -1.0

    def eval_helper_ask_dimension_question(
        self,
        query,
        gold_hypo,
        gold_workflow,
        gen_hypo,
        gen_workflow,
        dataset_meta,
        dimension,
        dataset_type,
        use_column_metadata=True,
    ):
        """Adapted from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
        `ask_dimension_question`."""
        dimension_question = ""
        score = 0.0
        if dimension == "var":
            score = {"p": -1.0, "r": -1.0, "f1": -1.0}

        prompt = "You are an AI assistant that helps evaluate a data-driven hypothesis. You are a helpful assistant who is not talkative. You only respond with the exact answer to a query without additional conversation."
        if dimension == "context":
            dimension_question = """\
            Question: Is HypoB defined in the same context as HypoA?
            (Context refers to assumptions/stratification under which the hypotheses are defined.)
            Options: A) same   B) different
            What is your answer?"""
        elif dimension == "var":
            dimension_question = """\
            Question: For both HypoA and HypoB, what are the different variables found in the hypotheses? \
            Return your answer as a JSON object in the following format:
            ```json
            {{
            "sizeA": num of variables used in HypoA
            "sizeB": num of variables used in HypoB
            "intersection": num of variables common in HypoA and HypoB. Use *fuzzy matching* to determine intersection, accounting for paraphrases or slightly different surface forms
            "explanation": a short text explanation about the variables
            }}```
            Answer:"""
        elif dimension == "rel":
            dimension_question = """\
            Question: Does HypoB exhibit the same relation as HypoA?
            Compare using following example hierarchy of relationships (based on specificity): \
            "there exists a relationship" > "positive relationship" > "positive AND (linear OR quadratic)" > "positive AND linear".
            Options: A) very similar B) similar but general than HypoA C) different
            Return your answer as a JSON object in the following format:
            ```json
            {{
            "answer": one of the options from A) very similar B) similar but general than HypoA C) different
            "explanation": a short text explanation about the relationship comparison
            }}```
            Answer:"""

        datasets_json = self.eval_helper_prepare_dataset_metadata_json(
            dataset_meta,
            dataset_type=dataset_type,
            use_column_metadata=use_column_metadata,
        )

        dimension_question_str = f"""\
            You are going to compare two natural-language hypotheses HypoA and HypoB accompanied with optional workflows: WorkflowA for HypoA and WorkflowB for HypoB. \
            Both the hypotheses answer the natural language query "QUERY" over the dataset(s) described by dataset description(s) and column description(s) below. \
            Compare HypoA and HypoB in terms of three aspects: Contexts, Variables, and Relations. \
            E.g., for the hypothesis "From 1995 to 2009, the number of sandhill cranes around the tundra (Indigilka River) surged by an astounding ~10X":
            * Contexts refer to stratification of the data under which the given hypothesis is True. E.g., "For all women", "From 1995 to 2009".
            * Variables refer to the set of variables (either dependent or independent) that are mentioned in the hypothesis. E.g., number of sandhill cranes, location.
            * Relations refer to the form of relation between the variables. E.g., "surged by ~10x".

            Answer following questions for a given pair of hypotheses, HypoA and HypoB, along with an explanation grounded on the QUERY and the DATASET(S).

            Here is the metadata for the task:
            ```json
            {{
            "datasets": {datasets_json},
            "query": {query},
            "HypoA": {gold_hypo},
            "WorkflowA": {gold_workflow},
            "HypoB": {gen_hypo},
            "WorkflowB": {gen_workflow}
            }}
            ```

            {dimension_question}"""

        prompt = "\n\n".join([prompt, dimension_question_str])
        answer = self.eval_helper_get_response_with_retry(
            prompt=prompt, decode_output_to_json=False, temperature=0, max_retry=1
        )

        if answer:
            score = self.eval_helper_get_score_from_answer(
                type=dimension, answer=answer
            )

        return dimension_question, answer, score

    def run_eval_gold_vs_gen_NL_subhypo(
        self,
        query,
        gold_hypo,
        gold_workflow,
        gen_hypo,
        gen_workflow,
        dataset_meta,
        context_score,
        dataset_type,
        use_column_metadata=True,
    ):
        """GPT-4 based evaluation to evaluate generated hypothesis in terms of context, variables, relation.

        This function is adapted from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
            `run_eval_gold_vs_gen_NL_subhypo`.
        """

        eval_rec = {
            "query": query,
            "HypoA": gold_hypo,
            "WorkflowA": gold_workflow,
            "HypoB": gen_hypo,
            "WorkflowB": gen_workflow,
        }

        for dimension in ["var", "rel"]:
            question, answer, score = self.eval_helper_ask_dimension_question(
                query=query,
                gold_hypo=gold_hypo,
                gold_workflow=gold_workflow,
                gen_hypo=gen_hypo,
                gen_workflow=gen_workflow,
                dataset_meta=dataset_meta,
                dimension=dimension,
                dataset_type=dataset_type,
                use_column_metadata=use_column_metadata,
            )
            eval_rec[dimension] = {
                "question": question,
                "answer": answer,
                "score": score,
            }

        eval_rec["context"] = context_score
        eval_rec["accuracy_score"] = (
            1.0
            * eval_rec["context"]["score"]
            * eval_rec["var"]["score"]["f1"]
            * eval_rec["rel"]["score"]
        )

        return eval_rec

    def run_eval_gold_vs_gen_NL_hypo(
        self,
        query,
        gold_hypo,
        gen_hypo,
        dataset_meta,
        dataset_type,
        use_column_metadata=True,
    ):
        """Evaluate the generated hypothesis against the gold hypothesis.

        This function is adapted from https://github.com/allenai/discoverybench/blob/main/eval/new_eval.py
            `run_eval_gold_vs_gen_NL_hypo_workflow`. Remove workflow because it is marked as optional in the
            original function.

        Input: Dataset Metadata, Query, Gold Hypothesis Hg, Pred Hypothesis Hg
        Output: eval_rec json includes final_score
        Procedure:
            1. Compute sub-hypotheses for Hg and Hp.
                Gold: [Hg1, Hg2] (compute on the fly)
                Predicted: [Hp1, Hp2] (compute on the fly)
            2. Compute Intersection: [(Hg_i, Hp_j), …]  # tuples of (gold,pred) that matched with context
            (do this w/o explicit extraction)
            3. Compute recall_context (programmatically)
                r_v_list = []
                For (Hg_i, Hp_j) in the intersection:
                    With Hg_i, Hp_j in NL, ask GPT4 → #variables and #intersection and a paragraph explanation
                    and programmatically calculate f1_v
                    Hg_i, Hp_j in NL, ask GPT4 → matching score (0, 0.5 or 1) :
                    A) very similar B) similar but general than HypoA C) different + explanation
                    r_v_list ← f1_v * score_r
                accuracy_score = mean(r_v_list)
                score = [ recall_context * mean over predicted context(context_score * var_score * rel_score)]
        """
        eval_rec = {
            "query": query,
            "HypoA": gold_hypo,
            "HypoB": gen_hypo,
        }

        gold_sub_hypo_json = self.eval_helper_get_sub_hypotheses(
            hypo=gold_hypo,
            dataset_meta=dataset_meta,
            use_column_metadata=use_column_metadata,
        )
        if len(gold_sub_hypo_json["sub_hypo"]) == 0:
            gold_sub_hypo_json["sub_hypo"] = [
                {
                    "text": gold_hypo,
                    "context": "None",
                    "variables": [],
                    "relations": "",
                    "explanation": "unable to segment",
                }
            ]
        print(f"gold_sub_hypo_json: {gold_sub_hypo_json}")

        gen_sub_hypo_json = self.eval_helper_get_sub_hypotheses(
            hypo=gen_hypo,
            dataset_meta=dataset_meta,
            use_column_metadata=use_column_metadata,
        )
        if len(gen_sub_hypo_json["sub_hypo"]) == 0:
            gen_sub_hypo_json["sub_hypo"] = [
                {
                    "text": gen_hypo,
                    "context": "None",
                    "variables": [],
                    "relations": "",
                    "explanation": "unable to segment",
                }
            ]
        print(f"gen_sub_hypo_json: {gen_sub_hypo_json}")

        eval_rec["gold_sub_hypo"] = gold_sub_hypo_json
        eval_rec["gen_sub_hypo"] = gen_sub_hypo_json

        gold_subh_covered = []
        gen_subh_to_gold_subh = dict()
        gen_gold_subh_to_context = dict()

        for p_id, gen_subh in enumerate(gen_sub_hypo_json["sub_hypo"]):
            gen_subh_to_gold_subh[p_id] = -1

            for g_id, gold_subh in enumerate(gold_sub_hypo_json["sub_hypo"]):
                if g_id in gold_subh_covered:
                    continue

                # match context
                context_bool = self.eval_helper_is_matching_context(
                    gold_subh["text"],
                    gold_subh.get("context", ""),
                    gen_subh["text"],
                    gen_subh.get("context", ""),
                )
                if context_bool:
                    context_score = 1.0
                else:
                    context_score = 0.0

                if context_score == 1.0:  # match only when context_score = 1.0
                    gen_subh_to_gold_subh[p_id] = g_id
                    gold_subh_covered.append(g_id)
                    gen_gold_subh_to_context[f"P{p_id}||G{g_id}"] = {
                        "question": f"""Comapring: GoldH: {gold_subh["text"]}, GoldC: {gold_subh['context']}\nGenH: {gen_subh['text']}, GenC: {gen_subh['context']}""",
                        "answer": context_bool,
                        "score": context_score,
                    }
                    break

        eval_rec["gen_subh_to_gold_subh"] = gen_subh_to_gold_subh
        eval_rec["gold_subh_covered"] = gold_subh_covered
        matched_gold_gen_subh_evals = dict()
        sum_accuracy_score = 0.0
        for p_id, g_id in gen_subh_to_gold_subh.items():
            if g_id >= 0:
                key = f"P{p_id}||G{g_id}"
                context_score = gen_gold_subh_to_context[key]
                subh_eval_rec = self.run_eval_gold_vs_gen_NL_subhypo(
                    query=query,
                    gold_hypo=gold_hypo,
                    gold_workflow="",
                    gen_hypo=gen_hypo,
                    gen_workflow="",
                    dataset_meta=dataset_meta,
                    context_score=context_score,
                    dataset_type=dataset_type,
                    use_column_metadata=use_column_metadata,
                )
                sum_accuracy_score += subh_eval_rec["accuracy_score"]
                matched_gold_gen_subh_evals[key] = subh_eval_rec

        eval_rec["matched_gold_gen_subh_evals"] = matched_gold_gen_subh_evals
        # Match whether the derived hypothsis and the ground truth hypothesis are entailed
        eval_rec["recall_context"] = (
            len(gold_subh_covered) / len(gold_sub_hypo_json["sub_hypo"])
            if len(gold_sub_hypo_json["sub_hypo"])
            else 0.0
        )

        return eval_rec

    def evaluate_task_performance(self) -> Dict:
        performance = {"outcome": self.result_editor.get_text(), "query": self.query}
        if len(self.result_editor.get_text()) == 0:
            performance["task_completion"] = 0
            performance["performance_rating"] = 0
            return performance
        else:
            performance["task_completion"] = 1
        if self.use_simulated_dataset:
            detailed_result = self.run_eval_gold_vs_gen_NL_hypo(
                query=self.query,
                gold_hypo=self.discovery_bench_gold_hypo,
                gen_hypo=self.result_editor.get_text(),
                dataset_meta=self.discovery_bench_metadata,
                dataset_type="real",
                use_column_metadata=True,
            )
            performance["detailed_result"] = detailed_result
            performance["performance_rating"] = detailed_result["recall_context"]

        return performance

    def __repr__(self):
        return (
            f"CoDiscoveryEnv("
            f"use_simulated_dataset={self.use_simulated_dataset}, query={self.query}, datasets={self.datasets})"
        )

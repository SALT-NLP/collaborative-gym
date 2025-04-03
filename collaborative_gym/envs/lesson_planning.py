import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from PyPDF2 import PdfReader
from knowledge_storm import GoogleSearch
import trafilatura

from collaborative_gym.core import CoEnv, ObservationTypes, logger
from collaborative_gym.envs.registry import EnvFactory
from collaborative_gym.spaces import (
    MultiSpace,
    MAX_UNICODE_LENGTH,
    UnicodeWithRegexPattern,
)
from collaborative_gym.utils.string import post_process_parsed_function_arg
from collaborative_gym.utils.text_editor import TextEditor

IM_MARKDOWN_DATAPOINTS = [
    (
        'Help me create a lesson plan for "Relationships between Quantities".',
        ["IM_u6l1_cleaned.md", "md"],
    ),
    (
        'Help me create a lesson plan for "The Distributive Property, Part 1".',
        ["IM_u6l9_cleaned.md", "md"],
    ),
    (
        'Help me create a lesson plan for "Positive and Negative Numbers".',
        ["IM_u7l1_cleaned.md", "md"],
    ),
]

IM_PDF_DATAPOINTS = [
    (
        'Help me create a lesson plan for "Relationships between Quantities".',
        ["IM_u6l1.pdf", "pdf"],
    ),
    (
        'Help me create a lesson plan for "The Distributive Property, Part 1".',
        ["IM_u6l9.pdf", "pdf"],
    ),
    (
        'Help me create a lesson plan for "Positive and Negative Numbers".',
        ["IM_u7l1.pdf", "pdf"],
    ),
]


def convert_to_markdown(source: str, source_type: str) -> str:
    """Convert different source types to markdown content.

    Args:
        source (str): Path to file or URL
        source_type (str): Type of source ('pdf', 'url', 'md')

    Returns:
        str: Content in markdown format

    Raises:
        ValueError: If source type is not supported or conversion fails
    """
    if source_type == "md":
        with open(source, "r", encoding="utf-8") as f:
            return f.read()

    elif source_type == "pdf":
        try:
            reader = PdfReader(source)
            text = "\n\n".join(page.extract_text() for page in reader.pages)
            # Basic formatting to markdown
            return f"# {os.path.basename(source)}\n\n{text}"

        except Exception as e:
            raise ValueError(f"Failed to convert PDF: {str(e)}")

    elif source_type == "url":
        try:
            downloaded = trafilatura.fetch_url(source)
            text = trafilatura.extract(
                downloaded, include_links=True, include_formatting=True
            )
            if not text:
                raise ValueError("No content could be extracted from URL")
            return text

        except Exception as e:
            raise ValueError(f"Failed to process URL: {str(e)}")

    else:
        raise ValueError(f"Unsupported source type: {source_type}")


class CoLessonPlanningActions(Enum):
    EDITOR_UPDATE = "EDITOR_UPDATE"
    FINISH = "FINISH"
    INTERNET_SEARCH = "INTERNET_SEARCH"

    def __str__(self):
        return self.value


MAX_CURRICULUM_LENGTH = 20000


@EnvFactory.register("lesson_planning")
class CoLessonPlanningEnv(CoEnv):
    """
    ## Description
    CoLessonPlanningEnv is a collaborative environment for creating lesson plan based on the given curriculum.
    The environment supports general internet search besides reading the curriculum and writing the lesson plan.

    ## Action Space
    Actions are strings that must match one of the following patterns:
    - EDITOR_UPDATE(text: str): Update editor content
    - INTERNET_SEARCH(query: str): Perform web search
    - FINISH(): Complete task

    ## Observation Space
    The observation is a dictionary containing:
    - lesson_plan_editor (non-private): Current state of the lesson plan
    - curriculum (non-private): Curriculum for the lesson planning
    - search_output (private): Results from recent search
    """

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        add_pedagogical_knowledge: bool = True,
        use_simulated_dataset: bool = False,
        simulated_dataset_data_point_idx: Optional[int] = None,  # [0, 1, 2]
        query: Optional[str] = None,
        curriculum_sources: Optional[
            List[tuple[str, str]]
        ] = None,  # List of (source, type) tuples
    ):
        super().__init__(team_members=team_members, env_id=env_id)

        self.use_simulated_dataset = use_simulated_dataset
        self.query = query
        self.curriculum_sources = curriculum_sources

        if self.use_simulated_dataset:
            self.query = IM_PDF_DATAPOINTS[simulated_dataset_data_point_idx][0]
            self.curriculum_sources = IM_PDF_DATAPOINTS[
                simulated_dataset_data_point_idx
            ][1]

        # Task information
        pedagogical_knowledge = (
            "A high-quality lesson plan includes:\n"
            "1. Integrating objectives for student learning.\n"
            "2. Incorporating teaching/learning activities.\n"
            "3. Embedding strategies to check student understanding.\n"
            "Some teaching strategies include:\n"
            "1. Gaining attention: Techniques to capture students' interest.\n"
            "2. Informing learners of objectives: Clearly stating what students will be able to do by the end of the lesson.\n"
            "3. Stimulating recall of prior learning: Helping students connect new information with what they already know.\n"
            "4. Presenting the content: Delivering the new information in various engaging ways.\n"
            "5. Providing learning guidance: Offering support and strategies to aid learning processes.\n"
            "6. Eliciting performance: Encouraging students to demonstrate their understanding.\n"
            "7. Providing feedback: Giving constructive feedback to enhance learning.\n"
            "8. Assessing performance: Measuring students' understanding to ensure learning objectives are met.\n"
            "9. Enhancing retention and transfer: Helping students apply what they learned to different contexts.\n"
            "Also, please add links to resources that can be used in the lesson plan if any.\n"
        )

        if add_pedagogical_knowledge:
            self.task_description = (
                "Your task is to create a lesson plan based on the given curriculum and query. "
                "Write the plan in the editor.\n"
                f"Pedagogical Knowledge: {pedagogical_knowledge}\n"
                f"Query: {self.query}\n"
            )
        else:
            self.task_description = (
                "Your task is to create a lesson plan based on the given curriculum and query. "
                "Write the plan in the editor.\n"
                f"Query: {self.query}"
            )
        self.example_question = ""
        self.example_trajectory = []

        # Use real search tools
        try:
            self.search_engine = GoogleSearch(
                google_search_api_key=os.environ["GOOGLE_SEARCH_API_KEY"],
                google_cse_id=os.environ["GOOGLE_CSE_ID"],
                k=10,
            )
        except KeyError:
            self.search_engine = None
            logger.error(
                "Please provide your Google API key in the environment variable GOOGLE_MAP_API_KEY and "
                "GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID to enable the search functionalities."
            )

        # Action Space
        self.action_space = MultiSpace(
            (
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(
                        r"^EDITOR_UPDATE\(text=(.*)\)$", re.DOTALL
                    ),
                    params=["text"],
                    machine_readable_identifier=CoLessonPlanningActions.EDITOR_UPDATE,
                    human_readable_name="Update the lesson plan editor",
                    human_readable_description="Update the lesson plan editor with the provided text. The full original"
                    " text will be replaced.",
                ),
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(r"^FINISH\(\)$", re.DOTALL),
                    params=[],
                    machine_readable_identifier=CoLessonPlanningActions.FINISH,
                    human_readable_name="Finish",
                    human_readable_description="Finish the lesson planning task.",
                ),
            )
        )

        self.private_action_space = MultiSpace(
            (
                UnicodeWithRegexPattern(
                    min_length=0,
                    max_length=MAX_UNICODE_LENGTH,
                    regex_pattern=re.compile(
                        r"^INTERNET_SEARCH\(query=(.*)\)$", re.DOTALL
                    ),
                    params=["query"],
                    machine_readable_identifier=CoLessonPlanningActions.INTERNET_SEARCH,
                    human_readable_name="Search the Internet",
                    human_readable_description="Search the Internet based on the provided query.",
                ),
            )
        )

        # Private observation
        self.search_output = {team_member: {} for team_member in team_members}

        # Shared observation
        self.lesson_plan_editor = TextEditor()
        curriculum = []
        if self.curriculum_sources:
            for source, source_type in self.curriculum_sources:
                curriculum.append(convert_to_markdown(source, source_type))
        self.curriculum = "\n-----\n".join(curriculum)[:MAX_CURRICULUM_LENGTH]

    # Private actions
    def _internet_search(self, role: str, query: str):
        output = self.search_engine(query)
        self.search_output[role] = {
            "query": f"Internet Search (query: {query})",
            "output": output,
        }

    # Shared actions
    def _editor_update(self, text: str):
        self.lesson_plan_editor.update_text(text)

    def close(self):
        pass

    def get_obs(self):
        return {
            "public": {
                "lesson_plan_editor": self.lesson_plan_editor.get_text(),
                "curriculum": self.curriculum,
            },
            "private": {
                team_member: {
                    "search_output": self.search_output[team_member],
                }
                for team_member in self.team_members
            },
        }

    def obs_type(self) -> dict[str, ObservationTypes]:
        return {
            "lesson_plan_editor": ObservationTypes.TEXT_EDITOR,
            "curriculum": ObservationTypes.MARKDOWN_READER,
            "search_output": ObservationTypes.SEARCH_INTERFACE,
        }

    def reset(
        self,
        options: dict[str, Any] | None = None,
    ):
        self.lesson_plan_editor.update_text("")
        self.search_output = {team_member: {} for team_member in self.team_members}

        return self.get_obs(), {}

    def step(self, role: str, action: str):
        """Execute one timestep within the environment.

        Args:
            role (str): The team member executing the action
            action (str): The action to take, formatted as a string matching one of the action space patterns

        Returns:
            observation (dict): Environment's current state
            reward (float): Amount of reward returned (-1 for errors, 0 otherwise)
            terminated (bool): Whether the episode has ended
            private (bool): Whether the action results are private to the acting agent
            info (dict): Contains auxiliary diagnostic information
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
        reward = 0  # Set intermediate reward to 0 if the action is successful; otherwise, -1.
        info["action_error"] = None
        try:
            if info["action"] == CoLessonPlanningActions.EDITOR_UPDATE:
                self._editor_update(text=parsed_action["text"])
            elif info["action"] == CoLessonPlanningActions.INTERNET_SEARCH:
                self._internet_search(role=role, query=parsed_action["query"])
            elif info["action"] == CoLessonPlanningActions.FINISH:
                terminated = True
        except Exception as e:
            err_msg = f"Error in executing the action: {action}. Error: {e}"
            return self.handle_action_error(err_msg, private)
        finally:
            info["action_end_time"] = time.time()

        # Get the observation
        obs = self.get_obs()

        return obs, reward, terminated, private, info

    def evaluate_task_performance(self) -> Dict:
        performance = {
            "outcome": self.lesson_plan_editor.get_text(),
            "query": self.query,
        }

        return performance

    def __repr__(self):
        return f"CoLessonPlanningEnv(use_simulated_dataset={self.use_simulated_dataset}, query={self.query})"

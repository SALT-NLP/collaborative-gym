"""Core API for human-agent collaboration environments and collaborative acts for team coordination."""

import re
import logging
from enum import Enum
from typing import Dict, SupportsFloat, Any, Tuple, Optional

from pydantic import BaseModel

from collaborative_gym.spaces import MultiSpace
from collaborative_gym.spaces import UnicodeWithRegexPattern, MAX_UNICODE_LENGTH

logging.basicConfig(
    level=logging.INFO, format="%(name)s : %(levelname)-8s : %(message)s"
)
logger = logging.getLogger(__name__)

ActType = str
ObsType = dict[str, Any]


class ObservationTypes(Enum):
    """Enumeration of observation rendering types for the GUI interface."""

    NO_RENDER = "NoRender"
    JUPYTER_NOTEBOOK = "JupyterEditor"
    TEXT_EDITOR = "TextEditor"
    PAPER_LIBRARY = "PaperLibrary"
    PAPER_SEARCH = "PaperSearchInterface"
    TRAVEL_SEARCH = "TravelSearchInterface"
    DISTANCE_MATRIX = "DistanceMatrix"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, ObservationTypes):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False


class CoEnv:
    """Base environment class for human-agent collaboration tasks.

    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the `step` and `reset` functions.
    An environment can be partially or fully observed by a single agent based on the private properties of observations.


    Attributes:
        task_description: Description of the collaborative task
        team_members: List of team member identifiers
        action_space: Actions that affect the shared environment
        private_action_space: Actions that only affect the acting member's environment
        additional_task_info: Optional task-specific information for creating information asymmetry
        example_question: Example task scenario for team members
        example_trajectory: Example sequence of actions and observations
        env_id: Unique identifier for this environment instance
    """

    task_description: str  # Description of the task
    team_members: list[str]  # List of team member identifiers
    action_space: MultiSpace  # Actions that will affect the shared environment
    private_action_space: MultiSpace  # Actions that will affect the private environment
    # Additional information about the task (can be used to create information asymmetry)
    additional_task_info: dict[str, Any] = {}
    # Associate each task environment with an example question and trajectory for team members to understand the task
    example_question: str = ""
    example_trajectory: list[Tuple[str, ActType, ObsType]] = (
        []
    )  # A sequence of Thought/Action/Observation tuples
    env_id: str  # Unique identifier for this environment instance

    def __init__(self, team_members: list[str], env_id: str):
        self.team_members = team_members
        self.env_id = env_id

    def step(
        self, role: str, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Returns:
            observation (ObsType): An element of the environment's observation_space as the current observation due to the agent actions.
            reward (SupportsFloat): The reward as a result of taking the action. 0 for success, -1 for failure unless otherwise specified.
            terminated (bool): Whether the agent reaches the terminal state.
            private (bool): Whether the change shall notify the action taker or be broadcast to all team members.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
        """
        raise NotImplementedError

    def handle_action_error(
        self, error_msg: str, private: bool = True
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Handle action errors in a consistent way across all environments.

        Args:
            error_msg (str): The error message to log and return
            private (bool): Whether this is a private action. Defaults to True for errors.

        Returns:
            tuple containing standard step() return values with error information
        """
        logging.error(error_msg)
        return self.get_obs(), -1, False, private, {"action_error": error_msg}

    def parse_and_validate_action(
        self, role: str, action: ActType
    ) -> tuple[dict[str, Any], bool, Optional[str], Optional[str]]:
        """Common routine for action parsing and validation.

        Args:
            role (str): The team member attempting the action
            action (ActType): The action string to parse and validate

        Returns:
            tuple containing:
                parsed_action (dict): Dictionary of parsed action parameters
                private (bool): Whether this is a private action
                action_id (Optional[str]): The machine readable identifier of the action if successful
                error_message (Optional[str]): Error message if validation fails, None if successful
        """
        # Validate role
        if role not in self.team_members:
            return {}, True, None, f"{role!r} is not a valid team member."

        # Validate action against action spaces
        private = False
        sanitized = action.strip()
        if self.private_action_space.contains(sanitized):
            private = True
        elif not self.action_space.contains(sanitized):
            # Minimal, defensive sanitation to handle edge cases like
            # "... Action\nEDITOR_UPDATE(text=...)" or extra prose around the action
            try:
                candidate = sanitized
                # Find the first ALL-CAPS function call, e.g., EDITOR_UPDATE(
                m = re.search(r"[A-Z][A-Z_]+\(", candidate)
                if m:
                    start = m.start()
                    end = candidate.rfind(")")
                    if end != -1 and end > start:
                        candidate = candidate[start : end + 1]
                    else:
                        candidate = candidate[start:]
                    # Drop any trailing "\nThought:" section if present
                    t_idx = candidate.find("\nThought:")
                    if t_idx != -1:
                        candidate = candidate[:t_idx].strip()
                    candidate = candidate.replace("\\(", "(").replace("\\)", ")")

                # Re-check spaces with the sanitized candidate
                if self.private_action_space.contains(candidate):
                    private = True
                    sanitized = candidate
                elif self.action_space.contains(candidate):
                    sanitized = candidate
                else:
                    # Fallback: try to locate any valid action by searching each action regex
                    best_match = None
                    best_private = False
                    # Search private actions first to preserve privacy semantics
                    for space, is_private in (
                        [(s, True) for s in self.private_action_space]
                        + [(s, False) for s in self.action_space]
                    ):
                        try:
                            # Remove anchors to allow in-string matching
                            pattern_str = space.pattern.pattern
                            pattern_str = pattern_str.lstrip('^').rstrip('$')
                            search_pat = re.compile(pattern_str, re.DOTALL)
                            m = search_pat.search(sanitized)
                            if m:
                                # Keep earliest match
                                if (best_match is None) or (m.start() < best_match.start()):
                                    best_match = m
                                    best_private = is_private
                        except Exception:
                            continue
                    if best_match is not None:
                        candidate2 = best_match.group(0).strip()
                        if best_private:
                            private = True
                        sanitized = candidate2
                    else:
                        return (
                            {},
                            True,
                            None,
                            f"{action!r} invalid. Please strictly follow the action space specifications.",
                        )
            except Exception:
                return (
                    {},
                    True,
                    None,
                    f"{action!r} invalid. Please strictly follow the action space specifications.",
                )

        # Parse action parameters using regex pattern
        action_space = self.private_action_space if private else self.action_space
        parsed_action = None
        for space in action_space:
            parsed_action = space.parse(sanitized)
            if parsed_action is not None:  # {} is also a valid parsed action
                return parsed_action, private, space.machine_readable_identifier, None

        # Should not reach here if action spaces are properly defined
        return {}, True, None, f"Failed to parse parameters from {action!r}"

    def reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info."""
        raise NotImplementedError

    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.

        This is critical for closing rendering windows, database or HTTP connections.
        Calling ``close`` on an already closed environment has no effect and won't raise an error.
        """
        pass

    def get_obs(self) -> ObsType:
        """Get the current observation of the environment.

        Returns:
            ObsType: Current observation dictionary containing the environment state.
        """
        raise NotImplementedError

    def obs_type(self) -> Dict[str, ObservationTypes]:
        """Return the type of each observation field for GUI rendering.

        Returns:
            Dict[str, ObservationTypes]: Mapping of observation keys to their rendering types.
        """
        raise NotImplementedError

    def action_space_to_description(self) -> str:
        """Convert the action space into a human-readable description.

        Returns:
            str: A string describing available actions and their parameters.
        """
        raise NotImplementedError

    def evaluate_task_performance(self) -> Dict:
        """Evaluate the performance of the team on the collaborative task.

        Returns:
            Dict: Performance metrics and statistics for the task completion.
        """
        pass

    def dump_action_space(self):
        """Dumps the action space to a JSON serializable format."""
        return [action.dump_json() for action in self.action_space] + [
            action.dump_json() for action in self.private_action_space
        ]

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        return f"<{type(self).__name__} instance>"

    def __repr__(self):
        """Gives a string representation of this environment."""
        pass

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False


class TeamMemberConfig(BaseModel):
    """
    Configuration class for team members in collaborative environments.

    Defines the properties and initialization parameters for both human and AI
    team members, including their type (agent, user, etc.), name, and startup
    configuration for node processes.

    Attributes:
        name: Unique identifier for the team member
        type: Type of team member ('agent', 'cmd_user', 'gui_user', etc.)
        start_node_base_command: Command used to start this member's node process
    """

    name: str
    type: str
    start_node_base_command: str


# Collaborative Actions
class SendTeammateMessage(UnicodeWithRegexPattern):
    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(
                r"^SEND_TEAMMATE_MESSAGE\(message=(.*)\)$", re.DOTALL
            ),
            params=["message"],
            machine_readable_identifier="SEND_TEAMMATE_MESSAGE",
            human_readable_name="Send a message to your teammate(s).",
            human_readable_description="Send a message to your teammate(s) to provide information, ask for feedback, "
            "allocate task, etc. This action is useful for collaboration.",
        )


class WaitTeammateContinue(UnicodeWithRegexPattern):
    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^WAIT_TEAMMATE_CONTINUE\(\)$", re.DOTALL),
            params=[],
            machine_readable_identifier="WAIT_TEAMMATE_CONTINUE",
            human_readable_name="Wait for your teammate(s) to continue.",
            human_readable_description="Skip your turn and wait for your teammate(s) to continue. This action is useful"
            " for collaboration, especially when you need to wait for your teammate(s) to "
            "provide information, complete certain parts before you can proceed, etc.",
        )


class RequestTeammateConfirm(UnicodeWithRegexPattern):
    """Request confirmation before the action is executed.

    This primitive action is useful to protect shared components in the shared environment.
    """

    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(
                r"^REQUEST_TEAMMATE_CONFIRM\(request_id=(.*), pending_action=(.*)\)$",
                re.DOTALL,
            ),
            params=["request_id", "pending_action"],
            machine_readable_identifier="REQUEST_TEAMMATE_CONFIRM",
            human_readable_name="Request confirmation from your teammate(s).",
            human_readable_description="For the pending action, request confirmation from your teammate(s) before "
            "executing the action.",
        )


class AcceptConfirmation(UnicodeWithRegexPattern):
    """Accept the confirmation request from the teammate."""

    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(
                r"^ACCEPT_CONFIRMATION\(request_id=(.*)\)$", re.DOTALL
            ),
            params=["request_id"],
            machine_readable_identifier="ACCEPT_CONFIRMATION",
            human_readable_name="Accept the confirmation request from your teammate(s).",
            human_readable_description="Accept the confirmation request from your teammate(s) for the pending action.",
        )


class RejectConfirmation(UnicodeWithRegexPattern):
    """Reject the confirmation request from the teammate."""

    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(
                r"^REJECT_CONFIRMATION\(request_id=(.*)\)$", re.DOTALL
            ),
            params=["request_id"],
            machine_readable_identifier="REJECT_CONFIRMATION",
            human_readable_name="Reject the confirmation request from your teammate(s).",
            human_readable_description="Reject the confirmation request from your teammate(s) for the pending action.",
        )


class PutAgentAsleep(UnicodeWithRegexPattern):
    """Put the agent to sleep.

    The action can be used by the human to control the usage of the agent.
    """

    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^PUT_AGENT_ASLEEP\(\)$", re.DOTALL),
            params=[],
            machine_readable_identifier="PUT_AGENT_ASLEEP",
            human_readable_name="Put the agent to sleep.",
            human_readable_description="Put the agent to sleep. After this action, the agent will not be notified of "
            "any new notifications and cannot take any actions until it is woken up.",
        )


class WakeAgentUp(UnicodeWithRegexPattern):
    """Wake the agent up.

    The action can be used by the human to control the usage of the agent.
    """

    def __init__(self):
        super().__init__(
            min_length=0,
            max_length=MAX_UNICODE_LENGTH,
            regex_pattern=re.compile(r"^WAKE_AGENT_UP\(\)$", re.DOTALL),
            params=[],
            machine_readable_identifier="WAKE_AGENT_UP",
            human_readable_name="Wake the agent up.",
            human_readable_description="Wake the agent up. After this action, the agent will be notified of new "
            "notifications and can take actions again.",
        )

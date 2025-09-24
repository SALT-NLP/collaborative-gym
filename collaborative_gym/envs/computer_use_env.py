"""Computer use environment for collaborative desktop automation tasks."""

import base64
import json
import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, SupportsFloat

from collaborative_gym.core import CoEnv, ObservationTypes
from collaborative_gym.envs.registry import EnvFactory
from collaborative_gym.spaces import MultiSpace, UnicodeWithRegexPattern, MAX_UNICODE_LENGTH
from collaborative_gym.utils.string import post_process_parsed_function_arg

from collaborative_gym.envs.computer_use.desktop_env.desktop_env import DesktopEnv
from collaborative_gym.envs.computer_use.observation import format_observation
from collaborative_gym.envs.computer_use.utils import parse_desktop_action, format_action_description

logger = logging.getLogger(__name__)


@EnvFactory.register("computer_use")
class CoComputerUseEnv(CoEnv):
    """Collaborative environment for desktop automation tasks using OSWorld.

    This environment enables human-agent teams to work together on computer-based
    tasks with shared screen visibility and collaborative actions.

    Supports two modes:
    - Real mode: Open-ended tasks with user-provided instructions
    - Simulated mode: Fixed tasks from OSWorld dataset with verifiable outcomes
    """

    def __init__(
        self,
        team_members: List[str],
        env_id: str,
        task_instruction: Optional[str] = None,
        use_simulated_dataset: bool = False,
        osworld_task_id: Optional[str] = None,
        osworld_dataset_path: str = "datasets/OSWorld",
        provider_name: str = "docker",
        headless: bool = True,
        require_a11y_tree: bool = False,
        require_terminal: bool = False,
        screen_size: tuple = (1920, 1080),
        **kwargs
    ):
        """Initialize the computer use environment.

        Args:
            team_members: List of team member identifiers
            env_id: Unique environment instance ID
            task_instruction: Task description for the team (ignored in simulated mode)
            use_simulated_dataset: Whether to use OSWorld evaluation dataset
            osworld_task_id: Task ID from dataset (e.g., "chrome/bb5e4c0d-f964-439c-97b6-bdb9747de3f4")
            osworld_dataset_path: Path to OSWorld evaluation examples
            provider_name: Virtualization provider (docker, vmware, etc.)
            headless: Whether to run in headless mode
            require_a11y_tree: Whether to include accessibility tree
            require_terminal: Whether to include terminal output
            screen_size: Desktop screen resolution
            **kwargs: Additional arguments for DesktopEnv
        """
        super().__init__(team_members=team_members, env_id=env_id)

        # Store simulated mode settings
        self.use_simulated_dataset = use_simulated_dataset
        self.osworld_dataset_path = osworld_dataset_path
        self.osworld_task_id = osworld_task_id
        self.task_config = None

        # Load task configuration if in simulated mode
        if self.use_simulated_dataset:
            if not osworld_task_id:
                raise ValueError("osworld_task_id is required when use_simulated_dataset is True")

            # Load the task configuration from dataset
            task_path = os.path.join(osworld_dataset_path, "examples", f"{osworld_task_id}.json")
            with open(task_path, "r") as f:
                self.task_config = json.load(f)

            # Use instruction from task config
            self.task_instruction = self.task_config["instruction"]
        else:
            # Use provided instruction or default
            self.task_instruction = task_instruction or "Complete computer tasks collaboratively."

        self.task_description = (
            "You are working with your team to complete computer-based tasks. "
            "You can control the desktop using various actions like clicking, typing, and dragging. "
            "The screen is shared among all team members, so coordinate your actions. "
            f"Task: {self.task_instruction}"
        )

        self.desktop_env = DesktopEnv(
            provider_name=provider_name,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            screen_size=screen_size,
            action_space="computer_13",
            **kwargs
        )

        # Initialize shared state
        self.screenshot = None
        self.action_history = []
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal

        # Define action space
        self.action_space = self._create_action_space()
        self.private_action_space = MultiSpace(())  # No private actions

        # Example trajectory for agents
        self.example_question = "Open a web browser and navigate to Google."
        self.example_trajectory = [
            (
                "I need to open a web browser. Let me click on the browser icon.",
                "CLICK(x=50, y=100)",
                {"screenshot": "[Browser opening]"}
            ),
            (
                "Now I'll type the Google URL in the address bar.",
                "TYPE(text='www.google.com')",
                {"screenshot": "[Browser with URL]"}
            ),
            (
                "Let me press Enter to navigate.",
                "KEY(key='enter')",
                {"screenshot": "[Google homepage]"}
            )
        ]

    def _create_action_space(self) -> MultiSpace:
        """Create the action space for desktop control."""
        return MultiSpace([
            # Click action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^CLICK\(x=(\d+\.?\d*),\s*y=(\d+\.?\d*)(?:,\s*button=(left|right|middle))?\)$'),
                params=["x", "y", "button"],
                machine_readable_identifier="CLICK",
                human_readable_name="Click at position",
                human_readable_description="Click at specified coordinates with optional button (left/right/middle)"
            ),
            # Type action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^TYPE\(text=(.*)\)$', re.DOTALL),
                params=["text"],
                machine_readable_identifier="TYPE",
                human_readable_name="Type text",
                human_readable_description="Type the specified text at current cursor position"
            ),
            # Key press action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^KEY\(key=(.*)\)$'),
                params=["key"],
                machine_readable_identifier="KEY",
                human_readable_name="Press key",
                human_readable_description="Press a keyboard key (e.g., enter, escape, tab)"
            ),
            # Scroll action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^SCROLL\(direction=(up|down|left|right),\s*amount=(\d+)\)$'),
                params=["direction", "amount"],
                machine_readable_identifier="SCROLL",
                human_readable_name="Scroll",
                human_readable_description="Scroll in specified direction by given amount"
            ),
            # Drag action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^DRAG\(start_x=(\d+\.?\d*),\s*start_y=(\d+\.?\d*),\s*end_x=(\d+\.?\d*),\s*end_y=(\d+\.?\d*)\)$'),
                params=["start_x", "start_y", "end_x", "end_y"],
                machine_readable_identifier="DRAG",
                human_readable_name="Drag",
                human_readable_description="Drag from start position to end position"
            ),
            # Move cursor action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^MOVE_TO\(x=(\d+\.?\d*),\s*y=(\d+\.?\d*)\)$'),
                params=["x", "y"],
                machine_readable_identifier="MOVE_TO",
                human_readable_name="Move cursor",
                human_readable_description="Move cursor to specified position without clicking"
            ),
            # Screenshot action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^SCREENSHOT\(\)$'),
                params=[],
                machine_readable_identifier="SCREENSHOT",
                human_readable_name="Take screenshot",
                human_readable_description="Capture current screen state"
            ),
            # Finish action
            UnicodeWithRegexPattern(
                min_length=0,
                max_length=MAX_UNICODE_LENGTH,
                regex_pattern=re.compile(r'^FINISH\(\)$'),
                params=[],
                machine_readable_identifier="FINISH",
                human_readable_name="Finish task",
                human_readable_description="Mark the task as completed"
            ),
        ])

    def step(
        self, role: str, action: str
    ) -> tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one timestep within the environment.

        Args:
            role: The team member executing the action
            action: The action string to execute

        Returns:
            observation: Current environment state
            reward: Reward value (0 for success, -1 for error)
            terminated: Whether episode has ended
            private: Whether action is private to actor
            info: Additional information
        """
        info = {"action_start_time": time.time()}

        # Parse and validate action
        parsed_action, private, action_id, err_msg = self.parse_and_validate_action(role, action)
        if err_msg:
            return self.handle_action_error(err_msg, private)

        # Process parameters
        for k in parsed_action:
            if parsed_action[k] is not None:
                parsed_action[k] = post_process_parsed_function_arg(parsed_action[k])

        info["action"] = action_id
        terminated = False
        reward = 0

        # Handle desktop actions
        if action_id == "FINISH":
            terminated = True

        elif action_id == "SCREENSHOT":
            if self.desktop_env:
                self.screenshot = self.desktop_env.controller.get_screenshot()

        else:
            # Convert action to OSWorld format and execute
            if self.desktop_env:
                desktop_action = parse_desktop_action(action)
                if desktop_action:
                    # Execute action in desktop environment
                    obs, _, _, _ = self.desktop_env.step(desktop_action)
                    self.screenshot = obs.get("screenshot")

                else:
                    return self.handle_action_error(f"Failed to parse desktop action: {action}", private)

        # Record action in history (just the action for completion checking)
        self.action_history.append(action)

        info["action_end_time"] = time.time()
        return self.get_obs(), reward, terminated, private, info

    def get_obs(self) -> Dict[str, Any]:
        """Get current observation of the environment."""
        # Get desktop observation if available
        desktop_obs = {}
        if self.desktop_env:
            desktop_obs = self.desktop_env._get_obs()
            self.screenshot = desktop_obs.get("screenshot")

        # Format observation
        return format_observation(
            screenshot=self.screenshot,
            team_members=self.team_members,
            accessibility_tree=desktop_obs.get("accessibility_tree") if self.require_a11y_tree else None,
            terminal_output=desktop_obs.get("terminal") if self.require_terminal else None
        )

    def obs_type(self) -> Dict[str, ObservationTypes]:
        """Return observation types for GUI rendering."""
        obs_types = {
            "screenshot": ObservationTypes.NO_RENDER,
        }

        if self.require_a11y_tree:
            obs_types["accessibility_tree"] = ObservationTypes.NO_RENDER
        if self.require_terminal:
            obs_types["terminal_output"] = ObservationTypes.NO_RENDER

        return obs_types


    def reset(self, options: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment to initial state."""
        # Reset shared state
        self.screenshot = None
        self.action_history = []

        # Reset desktop environment if available
        if self.desktop_env:
            # Use task config from dataset if in simulated mode
            if self.use_simulated_dataset and self.task_config:
                task_config = self.task_config
            else:
                task_config = options or {"id": "default", "instruction": self.task_instruction}

            desktop_obs = self.desktop_env.reset(task_config)
            self.screenshot = desktop_obs.get("screenshot")

        return self.get_obs(), {}

    def close(self):
        """Clean up environment resources."""
        if self.desktop_env:
            self.desktop_env.close()

    def evaluate_task_performance(self) -> Dict:
        """Evaluate task performance metrics.

        In simulated mode, uses the OSWorld evaluator from the task config.
        In real mode, relies on FINISH action or simple heuristics.
        """
        # Get OSWorld evaluation score if available
        task_score = 0.0
        if self.desktop_env and hasattr(self.desktop_env, 'evaluate') and hasattr(self.desktop_env, 'evaluator'):
            # OSWorld evaluate() returns float between 0.0 and 1.0
            # In simulated mode, this uses the evaluator from task_config
            task_score = float(self.desktop_env.evaluate())

        # Check if FINISH action was called
        finished = any(a == "FINISH()" for a in self.action_history)

        # Build metrics matching other environments' structure
        performance = {
            "outcome": base64.b64encode(self.screenshot).decode('utf-8') if self.screenshot else None,
            "query": self.task_instruction,
            "task_completion": 1 if (task_score >= 0.5 or finished) else 0,
            "performance_rating": task_score if task_score > 0 else (1.0 if finished else 0.0)
        }

        return performance
"""Observation formatting for the computer use environment."""

from typing import Dict, Any, List, Optional
import base64


def format_observation(
    screenshot: Optional[Any],
    last_action: Optional[str],
    task_instruction: str,
    team_members: List[str],
    accessibility_tree: Optional[str] = None,
    terminal_output: Optional[str] = None
) -> Dict[str, Any]:
    """Format observation for the computer use environment.

    Args:
        screenshot: Current desktop screenshot (can be numpy array or base64 string)
        last_action: Description of the last action taken
        task_instruction: Original task instruction
        team_members: List of team member names
        accessibility_tree: Optional accessibility tree data
        terminal_output: Optional terminal output

    Returns:
        Formatted observation dictionary with all public components
    """
    # Convert screenshot to base64 if needed
    screenshot_data = screenshot
    if screenshot is not None and hasattr(screenshot, 'shape'):
        # If it's a numpy array, convert to base64
        import cv2
        import numpy as np
        _, buffer = cv2.imencode('.png', screenshot)
        screenshot_data = base64.b64encode(buffer).decode('utf-8')

    # Build observation (all public, visible to all team members)
    obs = {
        "screenshot": screenshot_data,
        "last_action": last_action or "No action taken yet",
        "task_instruction": task_instruction
    }

    # Add optional components if available
    if accessibility_tree:
        obs["accessibility_tree"] = accessibility_tree
    if terminal_output:
        obs["terminal_output"] = terminal_output

    # Return observation in the standard format
    # Since there are no private actions, all observations are public
    return {
        "public": obs,
        "private": {member: {} for member in team_members}
    }


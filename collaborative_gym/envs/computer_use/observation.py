"""Observation formatting for the computer use environment."""

from typing import Dict, Any, List, Optional
import base64


def format_observation(
    screenshot: Optional[Any],
    team_messages: List[Dict[str, str]],
    last_action: Optional[str],
    task_instruction: str,
    team_members: List[str],
    accessibility_tree: Optional[str] = None,
    terminal_output: Optional[str] = None
) -> Dict[str, Any]:
    """Format observation for the computer use environment.

    Args:
        screenshot: Current desktop screenshot (can be numpy array or base64 string)
        team_messages: List of team communication messages
        last_action: Description of the last action taken
        task_instruction: Original task instruction
        team_members: List of team member names
        accessibility_tree: Optional accessibility tree data
        terminal_output: Optional terminal output

    Returns:
        Formatted observation dictionary with public and private components
    """
    # Convert screenshot to base64 if needed
    screenshot_data = screenshot
    if screenshot is not None and hasattr(screenshot, 'shape'):
        # If it's a numpy array, convert to base64
        import cv2
        import numpy as np
        _, buffer = cv2.imencode('.png', screenshot)
        screenshot_data = base64.b64encode(buffer).decode('utf-8')

    # Build public observation (visible to all team members)
    public_obs = {
        "screenshot": screenshot_data,
        "last_action": last_action or "No action taken yet",
        "task_instruction": task_instruction
    }

    # Build private observations for each team member
    private_obs = {}
    for member in team_members:
        member_obs = {
            "team_messages": [
                msg for msg in team_messages
                if msg.get("recipient") == member or msg.get("recipient") == "all"
            ]
        }

        # Add optional components if available
        if accessibility_tree:
            member_obs["accessibility_tree"] = accessibility_tree
        if terminal_output:
            member_obs["terminal_output"] = terminal_output

        private_obs[member] = member_obs

    return {
        "public": public_obs,
        "private": private_obs
    }


def extract_message_from_action(action: str) -> Optional[str]:
    """Extract message content from SEND_TEAMMATE_MESSAGE action.

    Args:
        action: Action string

    Returns:
        Message content if it's a send message action, None otherwise
    """
    import re
    match = re.match(r'^SEND_TEAMMATE_MESSAGE\(message=(.*)\)$', action, re.DOTALL)
    if match:
        return match.group(1).strip('"').strip("'")
    return None
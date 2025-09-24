"""Observation formatting for the computer use environment."""

from typing import Dict, Any, List, Optional
import base64


def format_observation(
    screenshot: Optional[Any],
    team_members: List[str],
    accessibility_tree: Optional[str] = None,
    terminal_output: Optional[str] = None
) -> Dict[str, Any]:
    """Format observation for the computer use environment.

    Args:
        screenshot: Current desktop screenshot (bytes from OSWorld)
        team_members: List of team member names
        accessibility_tree: Optional accessibility tree data
        terminal_output: Optional terminal output

    Returns:
        Formatted observation dictionary with all public components
    """
    # OSWorld always returns bytes, encode to base64
    screenshot_data = base64.b64encode(screenshot).decode('utf-8') if screenshot else None

    # Build observation with only workspace state
    obs = {
        "screenshot": screenshot_data,
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


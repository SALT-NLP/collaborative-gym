"""Utility functions for the computer use environment."""

import re
from typing import Dict, Any, Optional


def parse_desktop_action(action_str: str) -> Optional[Dict[str, Any]]:
    """Parse desktop action string into OSWorld-compatible format.

    Args:
        action_str: String representation of action (e.g., "CLICK(x=100, y=200)")

    Returns:
        Dictionary in OSWorld action format or None if parsing fails
    """
    # Extract action type and parameters
    match = re.match(r'^(\w+)\((.*)\)$', action_str)
    if not match:
        return None

    action_type = match.group(1)
    params_str = match.group(2)

    # Parse parameters
    params = {}
    if params_str:
        # Parse key=value pairs
        param_pairs = re.findall(r'(\w+)=([^,]+)', params_str)
        for key, value in param_pairs:
            # Try to convert to appropriate type
            value = value.strip().strip('"').strip("'")
            try:
                # Try integer first
                params[key] = int(value)
            except ValueError:
                try:
                    # Then float
                    params[key] = float(value)
                except ValueError:
                    # Keep as string
                    params[key] = value

    # Convert to OSWorld format based on action type
    if action_type == "CLICK":
        return {
            "action_type": "CLICK",
            "x": params.get("x", 0),
            "y": params.get("y", 0),
            "button": params.get("button", "left")
        }
    elif action_type == "TYPE":
        return {
            "action_type": "TYPE",
            "text": params.get("text", "")
        }
    elif action_type == "KEY":
        return {
            "action_type": "KEY",
            "key": params.get("key", "")
        }
    elif action_type == "SCROLL":
        return {
            "action_type": "SCROLL",
            "direction": params.get("direction", "down"),
            "amount": params.get("amount", 3)
        }
    elif action_type == "DRAG":
        return {
            "action_type": "DRAG",
            "start_x": params.get("start_x", 0),
            "start_y": params.get("start_y", 0),
            "end_x": params.get("end_x", 0),
            "end_y": params.get("end_y", 0)
        }
    elif action_type == "MOVE_TO":
        return {
            "action_type": "MOVE_TO",
            "x": params.get("x", 0),
            "y": params.get("y", 0)
        }
    elif action_type == "SCREENSHOT":
        return {
            "action_type": "SCREENSHOT"
        }

    return None


def format_action_description(action_type: str, params: Dict[str, Any]) -> str:
    """Format action into human-readable description.

    Args:
        action_type: Type of action
        params: Action parameters

    Returns:
        Human-readable action description
    """
    if action_type == "CLICK":
        button = params.get("button", "left")
        return f"Clicked {button} button at ({params.get('x', 0)}, {params.get('y', 0)})"
    elif action_type == "TYPE":
        return f"Typed: {params.get('text', '')}"
    elif action_type == "KEY":
        return f"Pressed key: {params.get('key', '')}"
    elif action_type == "SCROLL":
        return f"Scrolled {params.get('direction', 'down')} by {params.get('amount', 3)} units"
    elif action_type == "DRAG":
        return f"Dragged from ({params.get('start_x', 0)}, {params.get('start_y', 0)}) to ({params.get('end_x', 0)}, {params.get('end_y', 0)})"
    elif action_type == "MOVE_TO":
        return f"Moved cursor to ({params.get('x', 0)}, {params.get('y', 0)})"
    elif action_type == "SCREENSHOT":
        return "Captured screenshot"
    else:
        return f"Performed {action_type}"
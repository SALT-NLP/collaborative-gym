"""Prepare an offline, unlabelled review packet from a Co-Gym event_log.jsonl.

Run this file directly; no model, service, or installed Co-Gym package is needed.
The exporter selects candidate events, never judges their usefulness or uptake.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


def parse_event_log(text):
    """Read JSONL without silently discarding malformed or incomplete events."""
    events = []
    required = ("role", "action", "action_status", "action_type")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(event, dict) or any(
            not isinstance(event.get(key), str) for key in required
        ):
            raise ValueError(
                f"Line {line_number}: expected an event with string fields {required}"
            )
        events.append(event)
    return events


def build_review_packet(events, human_roles):
    """Retain the complete event sequence and create blank annotation records.

    Indices are zero-based event ordinals, not physical JSONL line numbers.
    Explicit, exact role matching avoids assuming that every role containing
    'user' is human. Failed events remain evidence but are not review anchors.
    Only the public part of current_observation is exported; no fallback to a
    private or unrecognised observation layout is made.
    """
    if isinstance(human_roles, str):
        raise ValueError("Pass human roles as a collection, not a single string")
    roles = set(human_roles)
    if not roles or any(
        not isinstance(role, str) or not role.strip() for role in roles
    ):
        raise ValueError("At least one non-empty human role is required")

    projected = []
    reviews = []
    for index, event in enumerate(events):
        observation = event.get("current_observation")
        public = observation.get("public") if isinstance(observation, dict) else None
        projected.append(
            {
                "event_index": index,
                "role": event["role"],
                "timestamp": event.get("timestamp"),
                "action": event["action"],
                "action_status": event["action_status"],
                "action_type": event["action_type"],
                "public_observation": copy.deepcopy(public),
            }
        )
        if event["role"] in roles and event["action_status"] == "succeeded":
            reviews.append(
                {
                    "contribution_event_index": index,
                    "contribution_span": None,
                    "disposition": None,
                    "evidence_event_indices": [],
                    "explanation": None,
                }
            )

    if not reviews:
        raise ValueError(
            "No successful events match the human roles; check the role names"
        )
    return {
        "schema_version": 1,
        "human_roles": sorted(roles),
        "event_count": len(events),
        "events": projected,
        "reviews": reviews,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument(
        "--human-role",
        action="append",
        required=True,
        help="Exact role; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.event_log.read_bytes()
        events = parse_event_log(raw.decode("utf-8"))
        packet = build_review_packet(events, args.human_role)
        packet["source_sha256"] = hashlib.sha256(raw).hexdigest()
        # Exclusive creation protects completed annotations and the source log.
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(packet, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()

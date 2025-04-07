import argparse
import json
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from task performance and event logs."
    )
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Directory containing experiment session folders.")
    return parser.parse_args()


def load_task_performance(session_dir):
    performance_path = os.path.join(session_dir, "task_performance.json")
    with open(performance_path) as f:
        return json.load(f)


def update_task_performance_if_needed(session_dir, task_performance):
    outcome = task_performance.get("outcome", "")
    if len(outcome.strip()) == 0:
        # Task not completed if outcome is empty.
        task_performance["task_completion"] = 0
        task_performance["performance_rating"] = 0
        performance_path = os.path.join(session_dir, "task_performance.json")
        with open(performance_path, "w") as f:
            json.dump(task_performance, f, indent=4)
    return task_performance


def process_event_log(session_dir):
    event_log_path = os.path.join(session_dir, "event_log.jsonl")
    events = []
    with open(event_log_path) as f:
        for line in f:
            events.append(json.loads(line))
    return events


def process_session(result_dir, session_name):
    session_dir = os.path.join(result_dir, session_name)
    task_performance = load_task_performance(session_dir)
    # task_performance = update_task_performance_if_needed(session_dir, task_performance)
    
    total_case = 1
    complete = task_performance.get("task_completion", 0)
    performance_rating = task_performance.get("performance_rating", 0)
    collaboration_score = complete * performance_rating

    # Load event log and count actions
    events = process_event_log(session_dir)
    agent_action_cnt = user_action_cnt = 0
    agent_collab_cnt = user_collab_cnt = 0

    for event in events:
        role = event.get("role", "")
        if "agent" in role:
            agent_action_cnt += 1
            if event.get("action_type") == "collaborative":
                agent_collab_cnt += 1
        elif "user" in role:
            user_action_cnt += 1
            if event.get("action_type") == "collaborative":
                user_collab_cnt += 1

    return {
        "complete": complete,
        "performance_rating": performance_rating,
        "collaboration_score": collaboration_score,
        "agent_action_cnt": agent_action_cnt,
        "user_action_cnt": user_action_cnt,
        "agent_collab_cnt": agent_collab_cnt,
        "user_collab_cnt": user_collab_cnt,
    }


def aggregate_sessions(result_dir):
    total_sessions = 0
    complete_list = []
    performance_list = []
    collab_scores = []
    agent_actions = []
    user_actions = []
    agent_msgs = []
    user_msgs = []

    for session_name in os.listdir(result_dir):
        session_path = os.path.join(result_dir, session_name)
        if not os.path.isdir(session_path):
            continue

        session_data = process_session(result_dir, session_name)
        total_sessions += 1
        complete_list.append(session_data["complete"])
        if session_data["complete"] == 1:
            performance_list.append(session_data["performance_rating"])
        collab_scores.append(session_data["collaboration_score"])
        agent_actions.append(session_data["agent_action_cnt"])
        user_actions.append(session_data["user_action_cnt"])
        agent_msgs.append(session_data["agent_collab_cnt"])
        user_msgs.append(session_data["user_collab_cnt"])

    if total_sessions == 0 or len(performance_list) == 0:
        print("No valid sessions found.")
        return

    aggregated_result = {
        "delivery_rate": sum(complete_list) / total_sessions,
        "task_performance": sum(performance_list) / len(performance_list),
        "collaboration_score": sum(collab_scores) / total_sessions,
        "average_agent_action_cnt": sum(agent_actions) / total_sessions,
        "avg_user_action_cnt": sum(user_actions) / total_sessions,
        "avg_agent_message_cnt": sum(agent_msgs) / total_sessions,
        "avg_user_message_cnt": sum(user_msgs) / total_sessions,
    }

    print(f"Delivery rate: {aggregated_result['delivery_rate']}")
    print(f"Task performance: {aggregated_result['task_performance']}")
    print(f"Collaboration score: {aggregated_result['collaboration_score']}")

    output_path = os.path.join(result_dir, "aggregated_result.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_result, f, indent=4)


def main():
    args = parse_arguments()
    aggregate_sessions(args.result_dir)


if __name__ == "__main__":
    main()

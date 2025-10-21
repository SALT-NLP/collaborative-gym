"""FastAPI server for the Collaborative Gym (Real).

This server is compatible with the Collaborative Gym frontend which
provides a shared workbench for users and agents to collaborate on tasks.
"""

import atexit
import json
import os
import signal
import sys
import time
import uuid
from threading import Thread
from typing import List

import numpy as np
import pandas as pd
import psutil
import redis
import toml
from aact import Message
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import (
    FastAPI,
    WebSocket,
    HTTPException,
    Request,
    Form,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from collaborative_gym.core import TeamMemberConfig, SendTeammateMessage
from collaborative_gym.nodes.base_node import LAST_ACTIVE_TIME_KEY
from collaborative_gym.nodes.commons import JsonObj
from collaborative_gym.nodes.gui_user import GUIUserListenNode
from collaborative_gym.runner import Runner
from collaborative_gym.utils.utils import load_api_key

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://co-gym.com", "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configs
load_api_key("secrets.toml")
DISABLE_AGENT = os.getenv("DISABLE_AGENT", "false").lower() == "true"

REDIS_URL = "redis://localhost:6379/0"
REDIS_CLIENT = redis.Redis.from_url(REDIS_URL)
REDIS_CLIENT.ping()
REDIS_ASYNC_CLIENT = redis.asyncio.from_url(REDIS_URL)
REDIS_ASYNC_CLIENT.ping()

# Local storage
SERVER_LOCAL_STORAGE_DIR = os.path.join(os.getcwd(), "workdir/server_local_storage")
if not os.path.exists(SERVER_LOCAL_STORAGE_DIR):
    os.makedirs(SERVER_LOCAL_STORAGE_DIR)
DOCKER_STORAGE_DIR = os.path.join(SERVER_LOCAL_STORAGE_DIR, "docker_local_storage")
if not os.path.exists(DOCKER_STORAGE_DIR):
    os.makedirs(DOCKER_STORAGE_DIR)

AGENT_NAME = "agent"  # Only one agent for now

runner = Runner(result_dir=SERVER_LOCAL_STORAGE_DIR, redis_url=REDIS_URL)


@app.websocket("/ws/{session_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    await websocket.accept()
    if not runner.check_session_exists(session_id):
        await websocket.close(code=4000, reason="Session not found")
        return

    async with GUIUserListenNode(
        env_uuid=f"env_{session_id}",
        node_name=f"user_{user_id}",
        team_members=[AGENT_NAME],
        websocket=websocket,
        redis_url=REDIS_URL,
    ) as user_listen_node:
        await user_listen_node.event_loop()


@app.post("/api/init_env")
async def init_environment(
    user_id: str = Form(...),
    env_class: str = Form(...),
    env_args: str = Form(...),
    files: List[UploadFile] = File(None),
):
    """Initialize a new session with env_class and env_args.

    Args:
        user_id (str): User ID.
        env_class (str): Environment class, need to match registered names in collaborative_gym.envs.
        env_args (str): Environment arguments in JSON format.
        files (List[UploadFile]): List of files (e.g., csv files for CoAnalysisEnv).
    """
    try:
        session_uuid = str(uuid.uuid4())
        env_args = json.loads(env_args)
        if env_class == "tabular_analysis":
            csv_files = []
            if files and len(files) > 0:
                os.makedirs(
                    os.path.join(SERVER_LOCAL_STORAGE_DIR, f"env_{session_uuid}"),
                    exist_ok=True,
                )
                for file in files:
                    file_path = os.path.join(
                        SERVER_LOCAL_STORAGE_DIR, f"env_{session_uuid}", file.filename
                    )
                    contents = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(contents)
                    csv_files.append(file_path)
            env_args["csv_files"] = csv_files
        env_config_path = os.path.join(
            SERVER_LOCAL_STORAGE_DIR, f"env_{session_uuid}_config.toml"
        )
        if env_class == "tabular_analysis":
            env_args["docker_local_root_dir"] = DOCKER_STORAGE_DIR
        with open(env_config_path, "w") as f:
            toml_string = toml.dumps({"env_class": env_class, "env_args": env_args})
            f.write(toml_string)
        if not DISABLE_AGENT:
            thread = Thread(
                target=runner.start_session,
                args=(
                    session_uuid,
                    env_config_path,
                    [
                        TeamMemberConfig(
                            name=f"user_{user_id}",
                            type="gui_user",
                            start_node_base_command="",
                        ),
                        TeamMemberConfig(
                            name=AGENT_NAME,
                            type="agent",
                            start_node_base_command="python -m "
                            "demo_agent.collaborative_agent_with_situational_planning.agent "
                            "--model-name gpt-4o --wait-time 1 --enhance-user-control",
                        ),
                    ],
                    100,  # max_steps (Set a large number to support long session)
                    False,
                    False,  # Not add tick node
                    120,  # Tick interval (not used)
                    30,  # Max tick count (not used)
                ),
            )
        else:
            thread = Thread(
                target=runner.start_session,
                args=(
                    session_uuid,
                    env_config_path,
                    [
                        TeamMemberConfig(
                            name=f"user_{user_id}",
                            type="gui_user",
                            start_node_base_command="",
                        ),
                    ],
                    100,  # max_steps (Set a large number to support long session)
                    False,
                    False,  # Not add tick node
                    120,  # Tick interval (not used)
                    30,  # Max tick count (not used)
                ),
            )
        thread.start()

        return {"message": "Environment initialized", "session_id": session_uuid}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/post_action/{session_id}/{user_id}")
async def post_user_action(request: Request, session_id: str, user_id: str):
    """Post user action to the environment.

    Args:
        request (Request): Request object, {"action": "valid action string"}.
        session_id (str): Session ID.
        user_id (str): User ID.
    """
    if not runner.check_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        data = await request.json()
        action_str = data["action"]
        payload = {
            "action": action_str,
            "role": user_id,
        }
        await REDIS_ASYNC_CLIENT.publish(
            f"env_{session_id}/step",
            Message[JsonObj](data=JsonObj(object=payload)).model_dump_json(),
        )  # publish must be async
        return {"status": "success", "message": f"Received action: {action_str}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/images/{session_id}/{file_name}")
async def get_image(session_id: str, file_name: str):
    """Get image file from the session directory.

    Used by the frontend to display images in Jupyter notebooks for CoAnalysisEnv.

    Args:
        session_id (str): Session ID.
        file_name (str): File name.
    """
    if not runner.check_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    file_path = os.path.join(DOCKER_STORAGE_DIR, f"env_{session_id}", file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return {"message": "File not found"}


@app.get("/api/tables/{session_id}")
async def get_tables(session_id: str):
    """Get CSV files from the session directory.

    Used by the frontend to display tables in for CoAnalysisEnv.

    Args:
        session_id (str): Session ID.
    """
    if not runner.check_session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    tables = []
    for file in os.listdir(os.path.join(SERVER_LOCAL_STORAGE_DIR, f"env_{session_id}")):
        if file.endswith(".csv"):
            try:
                file_path = os.path.join(
                    SERVER_LOCAL_STORAGE_DIR, f"env_{session_id}", file
                )
                df = pd.read_csv(file_path)
                # Change boolean columns to string to ensure display in the frontend
                for col in df.select_dtypes(include=[bool]).columns:
                    df[col] = df[col].astype(str)
                description = ""
                if len(df) > 1000:
                    description = (
                        f"Showing first 1000 rows out of {len(df)} rows. "
                        f"Please open the file on your local machine for the full content."
                    )
                    df = df.head(1000)
                records = df.replace({np.nan: None}).to_dict("records")

                # Create table entry matching the frontend schema
                table_data = {"name": file, "data": records, "description": description}
                tables.append(table_data)

            except Exception as e:
                print(f"Error reading CSV file {file}: {str(e)}")
                continue

    return tables


@app.get("/api/result/{session_id}")
async def get_result(session_id: str):
    """Get the result of the session.

    Args:
        session_id (str): Session ID.
    """
    try:
        event_log = []
        with open(
            os.path.join(
                SERVER_LOCAL_STORAGE_DIR, f"env_{session_id}", "event_log.jsonl"
            )
        ) as f:
            for line in f:
                event_log.append(json.loads(line))

        chat_history = []  # {"role, "message", "timestamp"}
        outcome_versions = []  # {"role", "chat_turn_id", "outcome"}
        send_message_action = SendTeammateMessage()
        for event in event_log:
            if (
                ("agent" in event["role"] or "user" in event["role"])
                and event["action_type"] == "collaborative"
                and event["action_status"] == "succeeded"
            ):
                action = event["action"]
                if send_message_action.contains(action):
                    message = send_message_action.parse(action)["message"]
                    chat_history.append(
                        {
                            "role": event["role"],
                            "message": message,
                            "timestamp": event["timestamp"],
                        }
                    )
            if event["action"].startswith("EDITOR_UPDATE(text="):  # Hard code for now
                outcome = event["action"].split("EDITOR_UPDATE(text=")[1].strip(')"')
                outcome = outcome.replace("\\n", "\n").strip().strip('"').strip("'")
                outcome_versions.append(
                    {
                        "role": event["role"],
                        "chat_turn_id": len(chat_history) - 1,
                        "outcome": outcome,
                    }
                )

        return {
            "outcome_versions": outcome_versions,
            "chat_history": chat_history,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


def kill_stale_model_processes(timeout_seconds=3600):
    """Kills stale model processes and marks the corresponding conversations as completed.

    Args:
        timeout_seconds (int): The number of seconds after which a process is considered stale.
    """
    print("Checking for stale processes...")

    def delete_process_record_in_session(pid):
        """Deletes the process record for a node from Redis.

        Args:
            pid (int): The process ID.
        """
        REDIS_CLIENT.hdel(LAST_ACTIVE_TIME_KEY, str(pid))

    if not REDIS_CLIENT.exists(LAST_ACTIVE_TIME_KEY):
        return

    for pid in REDIS_CLIENT.hkeys(LAST_ACTIVE_TIME_KEY):
        pid = pid.decode("utf-8")
        last_active_time = float(REDIS_CLIENT.hget(LAST_ACTIVE_TIME_KEY, pid))

        if time.time() - last_active_time > timeout_seconds:
            pid = int(pid)
            if pid == os.getpid():
                continue  # Skip the current process
            try:
                os.kill(
                    pid, signal.SIGTERM
                )  # Send SIGTERM instead of forcefully terminating
            except ProcessLookupError:
                # Process already terminated
                delete_process_record_in_session(pid)
                continue

            # Wait for the process to terminate gracefully
            time.sleep(5)
            try:
                corresponding_process = psutil.Process(pid)
                if corresponding_process.is_running():
                    corresponding_process.kill()  # Force kill if still running
            except psutil.NoSuchProcess:
                pass  # Process has already terminated

            delete_process_record_in_session(pid)
            print(f"Killed stale process for {pid}")
    print("Done checking for stale processes")


def handle_exit_signal(signum, frame):
    runner.cleanup_subprocesses()
    for pid in REDIS_CLIENT.hkeys(LAST_ACTIVE_TIME_KEY):
        pid = pid.decode("utf-8")
        REDIS_CLIENT.hdel(LAST_ACTIVE_TIME_KEY, pid)
    sys.exit(0)


# schedule a job to kill stale model processes
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=kill_stale_model_processes,
    trigger="interval",
    minutes=10,
)
scheduler.start()

signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)
atexit.register(runner.cleanup_subprocesses)
atexit.register(scheduler.shutdown)

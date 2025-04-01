from flask import Flask, request, jsonify
import subprocess
import os
import signal
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables all CORS

# Global variable to track the current running process
current_process = None

@app.route("/start-analysis", methods=["POST"])
def start_analysis():
    global current_process

    data = request.get_json()
    analysis_type = data.get("type")

    if current_process is not None:
        current_process.terminate()
        current_process = None

    if analysis_type == "pose":
        print("Starting pose detection stream on http://localhost:8002/video")
        current_process = subprocess.Popen(["python", "analyze_pose.py"])
    elif analysis_type == "shot":
        print("Starting shot tracking stream on http://localhost:8002/video")
        current_process = subprocess.Popen(["python", "analyze_shot.py"])
    else:
        return jsonify({"status": "error", "message": "Invalid type"}), 400

    return jsonify({"status": "ok"})


@app.route("/stop-analysis", methods=["POST"])
def stop_analysis():
    global current_process

    if current_process is not None:
        current_process.terminate()
        current_process = None
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"status": "no_process"})


if __name__ == "__main__":
    app.run(port=8001)

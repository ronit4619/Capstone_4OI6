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
        current_process = subprocess.Popen([
            "python", "analyze_pose.py",
            "--arm", data.get("arm", "default"),
            "--save", str(data.get("save_frames", False))
            ])
    elif analysis_type == "shot-release":
        print("Starting shot tracking stream on http://localhost:8002/video")
        current_process = subprocess.Popen([
            "python", "analyze_shot_release.py",
            "--arm", data.get("arm", "default"),
            ])
    elif analysis_type == "dribble-counter":
        print("Starting dribble counter stream on http://localhost:8002/video")
        current_process = subprocess.Popen([
            "python", "analyze_dribble.py"
            ])
    elif analysis_type == "jumpshot-analysis":
        # Jumpshot is handled separately via /upload-jumpshot
        print("Jumpshot analysis already started via upload route.")
    else:
        return jsonify({"status": "error", "message": "Invalid type"}), 400

    return jsonify({"status": "ok"})

@app.route("/upload-jumpshot", methods=["POST"])
def upload_jumpshot():
    global current_process

    if current_process is not None:
        current_process.terminate()
        current_process = None

    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video uploaded"}), 400

    file = request.files['video']
    arm = request.form.get("arm", "right")

    video_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(video_path)

    print(f"✅ Uploaded video saved to: {video_path}")

    current_process = subprocess.Popen([
        "python", "analyze_jumpshot.py",
        "--video", video_path,
        "--arm", arm
    ])

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

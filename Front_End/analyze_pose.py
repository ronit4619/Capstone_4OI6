from flask import Flask, Response
import cv2
import atexit

app = Flask(__name__)

# Open the default webcam
cap = cv2.VideoCapture(0)

# ✅ Ensure camera is released when the app exits
def cleanup():
    print("Releasing camera...")
    cap.release()

atexit.register(cleanup)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Dummy text overlay
        cv2.putText(frame, "Pose Detection Active", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting pose detection stream on http://localhost:8002/video")
    app.run(host='0.0.0.0', port=8002)

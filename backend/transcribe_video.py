from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os

app = Flask(__name__)
CORS(app)

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = "uploaded_video.mp4"  # Save the uploaded video to a specific file
    video_file.save(video_path)

    model = whisper.load_model("base")
    try:
        result = model.transcribe(video_path, word_timestamps=True)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Optionally, remove the uploaded video file after processing
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    app.run(debug=True)

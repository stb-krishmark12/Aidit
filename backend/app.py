# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import whisper
from moviepy.editor import VideoFileClip, concatenate_videoclips
import rembg
import numpy as np
from PIL import Image
import os
import json
import logging
import uuid
from fpdf import FPDF
import traceback

from images_to_video import VideoCreator
from video_to_images import ImageCreator
from main import segment_video

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

# Base upload directory
BASE_UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")

# Specific directories for features
FEATURE_DIRS = {
    "transcribe": os.path.join(BASE_UPLOAD_DIR, "transcribe"),
    "remove_silence": os.path.join(BASE_UPLOAD_DIR, "remove_silence"),
    "remove_bg_image": os.path.join(BASE_UPLOAD_DIR, "remove_bg_image"),
    "video_background_removal": os.path.join(BASE_UPLOAD_DIR, "video_background_removal")
}

# Create directories for each feature
for feature, base_dir in FEATURE_DIRS.items():
    os.makedirs(os.path.join(base_dir, "input"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)

@app.route('/transcribe', methods=['POST'])
def transcribe_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = os.path.join(FEATURE_DIRS['transcribe'], "input", "uploaded_video.mp4")
    video_file.save(video_path)

    model = whisper.load_model("base")
    try:
        result = model.transcribe(video_path, word_timestamps=True)
        
        # Save transcription as JSON
        transcription_path = os.path.join(FEATURE_DIRS['transcribe'], "results", "transcription.json")
        with open(transcription_path, 'w') as f:
            json.dump(result, f)

        # Extract text for frontend display
        transcription_text = " ".join(segment['text'] for segment in result['segments'])

        # Convert transcription to PDF
        pdf_path = os.path.join(FEATURE_DIRS['transcribe'], "results", "transcription.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for segment in result['segments']:
            pdf.cell(200, 10, txt=segment['text'], ln=True)
        pdf.output(pdf_path)

        return jsonify({"transcription": transcription_text, "pdf": "transcription.pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/download/pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    directory = os.path.join(BASE_UPLOAD_DIR, "transcribe", "results")
    return send_from_directory(directory, filename, as_attachment=True)

@app.route('/remove_silences', methods=['POST'])
def remove_silences():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = os.path.join(FEATURE_DIRS['remove_silence'], "input", "uploaded_video.mp4")
    video_file.save(video_path)

    output_video_path = os.path.join(FEATURE_DIRS['remove_silence'], "results", "no_silence.mp4")

    try:
        # Transcribe the video
        model = whisper.load_model("base")
        result = model.transcribe(video_path, word_timestamps=True)

        # Extract silence periods
        with VideoFileClip(video_path) as video:
            video_duration = video.duration
            silence_periods = identify_silence_periods(result, video_duration)

        # Remove silences from the video
        cut_silences(video_path, output_video_path, silence_periods)

        return jsonify({"output_video": "no_silence.mp4"})
    except Exception as e:
        logging.error(f"Error during silence removal: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/remove_background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_path = os.path.join(FEATURE_DIRS['remove_bg_image'], "input", "uploaded_image.png")
    image_file.save(image_path)

    try:
        input_image = Image.open(image_path)
        input_array = np.array(input_image)
        output_array = rembg.remove(input_array)
        output_image = Image.fromarray(output_array)
        output_path = os.path.join(FEATURE_DIRS['remove_bg_image'], "results", "no_bg.png")
        output_image.save(output_path)

        return jsonify({"output_image": "no_bg.png"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

@app.route('/video_background_removal', methods=['POST'])
def video_background_removal():
    if 'video' not in request.files:
        logging.error("No video file provided.")
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    video_path = os.path.join(FEATURE_DIRS['video_background_removal'], "input", "uploaded_video.mp4")
    video_file.save(video_path)
    logging.info(f"Video saved to: {video_path}")

    try:
        # Define paths for temporary images and output video
        temp_images_dir = os.path.join(FEATURE_DIRS['video_background_removal'], "input", "temp_images")
        output_path = os.path.join(FEATURE_DIRS['video_background_removal'], "results", "green_screen.mp4")

        # Create directory for temporary images
        os.makedirs(temp_images_dir, exist_ok=True)
        logging.info(f"Temporary images directory created at: {temp_images_dir}")

        # Extract frames from the video
        vid_to_im = ImageCreator(video_path, temp_images_dir, image_start=0, image_end=0)
        logging.info("Extracting frames from the video...")
        vid_to_im.get_images()

        # Call the segment_video function to process the frames
        logging.info("Calling segment_video function...")
        segment_video(
            video_filename=video_path,
            dir_frames=temp_images_dir,
            image_start=0,
            image_end=0,
            bbox_file=None,  # No bounding box file
            skip_vid2im=False,
            mobile_sam_weights="./models/mobile_sam.pt",
            auto_detect=True,  # Automatically detect without bounding box
            output_video=output_path,
            output_dir=os.path.join(FEATURE_DIRS['video_background_removal'], "input", "temp_processed_images"),
            pbar=False,
            reverse_mask=True,
        )

        logging.info(f"Video background removal completed. Output video path: {output_path}")
        return jsonify({"output_video": "green_screen.mp4"})
    except Exception as e:
        logging.error(f"Error in video background removal: {str(e)}")
        logging.error(traceback.format_exc())  # Log the full traceback
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    directory = os.path.join(BASE_UPLOAD_DIR, "video_background_removal", "results")
    return send_from_directory(directory, filename, as_attachment=True)

def identify_silence_periods(transcription, video_duration, threshold=1.0, buffer=0.1):
    silence_periods = []
    words = transcription['segments']
    previous_end = 0
    for word in words:
        start_time = word['start']
        if start_time - previous_end > threshold:
            silence_periods.append((previous_end + buffer, start_time - buffer))
        previous_end = word['end']
    if video_duration - previous_end > threshold:
        silence_periods.append((previous_end + buffer, video_duration - buffer))
    return silence_periods

def cut_silences(input_video, output_video, silence_periods):
    video = VideoFileClip(input_video)
    clips = []
    last_end = 0
    for (start, end) in silence_periods:
        if last_end < start:
            clips.append(video.subclip(last_end, start))
        last_end = end
    if last_end < video.duration:
        clips.append(video.subclip(last_end, video.duration))
    if clips:
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
    else:
        video.write_videofile(output_video, codec="libx264", audio_codec="aac")

if __name__ == '__main__':
    app.run(debug=True)

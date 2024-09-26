import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import time
from footballanalysis.yolodetector import YOLO
from footballanalysis.analysis import analyze_video, calculate_metrics

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to delete existing video file if it exists
def delete_existing_video(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# Function to download video using yt-dlp
def download_video(youtube_url, output_path):
    delete_existing_video(output_path)
    command = f"yt-dlp -o {output_path} {youtube_url}"
    subprocess.run(command, shell=True)

# Function to wait for the video file to be fully downloaded
def wait_for_file(file_path, timeout=60):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            st.error("Error: Timeout waiting for video file to be downloaded.")
            return False
        time.sleep(1)
    return True

st.title("Football Analysis Project")

# Upload video or provide URL
video_source = st.selectbox("Select Video Source", ["Upload", "Livestream URL", "Recorded URL", "YouTube URL"])

video_path = None

if video_source == "Upload":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        video_path = os.path.join(current_dir, "Include", "footballanalysis", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
elif video_source in ["Livestream URL", "Recorded URL", "YouTube URL"]:
    video_url = st.text_input("Enter the video URL")

if st.button("Analyze Video"):
    if video_source == "YouTube URL" and video_url:
        st.write("Downloading video...")
        output_path = os.path.join(current_dir, "Include", "footballanalysis", "video.mp4")
        download_video(video_url, output_path)
        if wait_for_file(output_path):
            video_path = output_path
    elif video_source == "Upload" and video_file:
        video_path = os.path.join(current_dir, "Include", "footballanalysis", video_file.name)

    if video_path:
        #st.write("Video Path:", video_path)
        st.write("Analyzing video...")
        #metrics = analyze_video(video_path)
        #st.write(metrics)
    else:
        st.error("Error: Video path is not set.")

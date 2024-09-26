import cv2
import numpy as np
from footballanalysis.yolodetector import YOLO
import os
import streamlit as st

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def analyze_video(video_path="video.mp4"):
    cfg_path = os.path.join(current_dir, "yolov3.cfg")
    weights_path = os.path.join(current_dir, "yolov3.weights")
    names_path = os.path.join(current_dir, "coco.names")

    st.write("CFG Path:", cfg_path)
    st.write("Weights Path:", weights_path)
    st.write("Names Path:", names_path)

    # Check if the video file exists
    video_path = os.path.join(current_dir, video_path)
    st.write("Video Path:", video_path)
    if not os.path.exists(video_path):
        st.error("Error: Video file does not exist.")
        return

    yolo = YOLO(cfg_path, weights_path, names_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    player_positions = []
    ball_positions = []

    frame_placeholder = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        st.write("Processing frame:", frame_count)

        # Process every 5th frame
        if frame_count % 5 == 0:
            outs = yolo.detect(frame)
            st.write("Detections:", outs)
            player_positions, ball_positions = process_detections(outs, frame, player_positions, ball_positions)

            # Display the frame in Streamlit
            frame_placeholder.image(frame, channels="BGR")

        frame_count += 1

    cap.release()

    metrics = calculate_metrics(player_positions, ball_positions)
    return metrics

def process_detections(outs, frame, player_positions, ball_positions):
    height, width, _ = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                if class_id == 0:  # Person class
                    player_positions.append((center_x, center_y))
                elif class_id == 32:  # Sports ball class
                    ball_positions.append((center_x, center_y))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return player_positions, ball_positions

def calculate_metrics(player_positions, ball_positions):
    # Calculate distance covered by each player
    distances = calculate_distances(player_positions)
    # Calculate ball possession
    possession = calculate_possession(ball_positions)
    # Calculate percentage of play in different parts of the field
    play_percentage = calculate_play_percentage(player_positions)

    metrics = {
        "Distances Covered": distances,
        "Ball Possession": possession,
        "Play Percentage": play_percentage
    }
    return metrics

def calculate_distances(player_positions):
    distances = {}
    for i in range(1, len(player_positions)):
        distance = np.linalg.norm(np.array(player_positions[i]) - np.array(player_positions[i - 1]))
        distances[i] = distance
    return distances

def calculate_possession(ball_positions):
    possession = {"Team A": 0, "Team B": 0}
    for position in ball_positions:
        if position in team_a_positions:
            possession["Team A"] += 1
        elif position in team_b_positions:
            possession["Team B"] += 1
    total = possession["Team A"] + possession["Team B"]
    if total > 0:
        possession["Team A"] = (possession["Team A"] / total) * 100
        possession["Team B"] = (possession["Team B"] / total) * 100
    else:
        possession["Team A"] = 0
        possession["Team B"] = 0
    return possession

def calculate_play_percentage(player_positions):
    play_percentage = {"Left": 0, "Center": 0, "Right": 0}
    for position in player_positions:
        if position[0] < width / 3:
            play_percentage["Left"] += 1
        elif position[0] > 2 * width / 3:
            play_percentage["Right"] += 1
        else:
            play_percentage["Center"] += 1
    total = sum(play_percentage.values())
    if total > 0:
        for key in play_percentage:
            play_percentage[key] = (play_percentage[key] / total) * 100
    return play_percentage

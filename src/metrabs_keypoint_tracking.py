import tensorflow as tf
import os
import cv2
import numpy as np
import csv

# Step 1: Download the Metrabs model
def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}_20211019.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

# Load the Metrabs model
def load_model(model_type='metrabs_mob3l_y4t'):
    model_path = download_model(model_type)
    model = tf.saved_model.load(model_path)
    return model

# Step 2: Process video and detect keypoints
def process_video(model, video_path, output_video_path, csv_output_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get video details
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the output video and CSV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Write header for CSV file
    with open(csv_output_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'Keypoint Index', 'X', 'Y'])  # Header row

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor and perform pose detection
        image = tf.convert_to_tensor(frame)
        pred = model.detect_poses(image, skeleton='smpl_24')

        # Annotate the frame with pose results
        annotated_frame = frame.copy()

        # Save keypoints to CSV and annotate the frame
        with open(csv_output_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for pose_index, pose2d in enumerate(pred['poses2d'].numpy()):
                for keypoint_index, (x, y) in enumerate(pose2d):  # Each joint is (x, y)
                    # Save to CSV: Frame, Keypoint Index, X, Y
                    csvwriter.writerow([frame_count, keypoint_index, x, y])
                    # Annotate the keypoint on the frame
                    cv2.circle(annotated_frame, (int(x), int(y)), 2, (0, 0, 255), -1)

        # Write the annotated frame to the output video
        out.write(annotated_frame)
        frame_count += 1

    # Release video and file objects
    cap.release()
    out.release()

    print(f"Output video saved at: {output_video_path}")
    print(f"Keypoints CSV saved at: {csv_output_path}")

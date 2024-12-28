import os
import numpy as np
from src.tapir_keypoint_tracking import TapirKeypointTracking

# Define paths
checkpoint_path = 'data/tapir_checkpoint_panning.npy'
input_folder = 'data/cello/input_video/'  # Folder containing input videos
output_folder = 'result/cello/output_video/'  # Folder for saving output videos
keypoints_folder = 'result/cello/output_csv/'  # Folder for saving keypoints CSV files

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(keypoints_folder, exist_ok=True)

# Initialize TAPIR keypoint tracking
tapir_tracker = TapirKeypointTracking(checkpoint_path)

# Define a mapping of video file names to their respective keypoints
video_keypoints_map = {
    "cello01.avi": np.array([(0, 183, 106)], dtype=np.float32),  # Example keypoints for video1
    "cello02.mp4": np.array([(0, 209, 86)], dtype=np.float32),  # Example keypoints for video2
    # Add more mappings for other videos
}

# Process all videos in the input folder
for video_file in os.listdir(input_folder):
    video_path = os.path.join(input_folder, video_file)

    # Check if the file is a valid video file
    if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Skipping non-video file: {video_file}")
        continue

    # Retrieve the keypoints for the video
    if video_file not in video_keypoints_map:
        print(f"No keypoints defined for video: {video_file}. Skipping.")
        continue

    keypoints = video_keypoints_map[video_file]

    # Define output paths for the video and keypoints CSV
    output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_keypoints.mp4")
    keypoints_file_path = os.path.join(keypoints_folder, f"{os.path.splitext(video_file)[0]}_keypoints.csv")

    # Run keypoint tracking on the video
    print(f"Processing video: {video_file} with keypoints: {keypoints}")
    tapir_tracker.track_keypoints(video_path, keypoints, output_video_path, keypoints_file_path)

print("Processing complete. Check the output folders for results.")

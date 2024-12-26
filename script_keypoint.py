import numpy as np
from src.tapir_keypoint_tracking import TapirKeypointTracking

# Define paths
checkpoint_path = '/content/drive/MyDrive/Violin/tapir_checkpoint_panning.npy'
video_path = '/content/drive/MyDrive/Violin/segmentation/cello_trial.avi'
output_video_path = '/content/drive/MyDrive/Violin/segmentation/bow_keypoints.mp4'
keypoints_file_path = '/content/drive/MyDrive/Violin/segmentation/bow_keypoints.csv'

# Initialize TAPIR keypoint tracking
tapir_tracker = TapirKeypointTracking(checkpoint_path)

# Define the keypoints of the violin (downsampled coordinates)
violin_keypoints = np.array([(0, 183, 106)], dtype=np.float32) #(0,y,x)

# Track keypoints and save outputs
tapir_tracker.track_keypoints(video_path, violin_keypoints, output_video_path, keypoints_file_path)

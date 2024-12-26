import os
from src.metrabs_keypoint_tracking import load_model, process_video

def main():
    # Define the paths for input and output files
    video_path = '/content/drive/MyDrive/Violin/segmentation/cello_metrabs.mp4'  # Replace with your video path
    output_video_path = '/content/drive/MyDrive/Violin/segmentation/metrabs_output.mp4'
    csv_output_path = '/content/drive/MyDrive/Violin/segmentation/keypoints.csv'

    # Load the model
    model = load_model()

    # Process the video and track keypoints
    process_video(model, video_path, output_video_path, csv_output_path)

if __name__ == '__main__':
    main()

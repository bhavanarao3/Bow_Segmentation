from src.visualization import annotate_video

# File paths
base_video_path = '/content/drive/MyDrive/Violin/segmentation/cello_trial.avi'  # Replace with your base video path
output_video_path = '/content/drive/MyDrive/Violin/segmentation/combined_video.mp4'
movement_csv = '/content/drive/MyDrive/Violin/segmentation/annotated.csv'

# Call the function to annotate the video
annotate_video(base_video_path, movement_csv, output_video_path)

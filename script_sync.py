from src.frame_sync import annotate_frames

# File paths
events_csv_path = '/content/drive/MyDrive/Violin/segmentation/pitch_detection_project_inferred_notes_with_silence.csv'
frames_csv_path = '/content/drive/MyDrive/Violin/segmentation/bow_movement_direction.csv'
output_csv_path = '/content/drive/MyDrive/Violin/segmentation/annotated.csv'

# Call the function to process and annotate frames
annotate_frames(events_csv_path, frames_csv_path, output_csv_path)

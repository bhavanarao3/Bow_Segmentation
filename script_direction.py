# script_movement.py

from src.bow_direction_pca import process_movement

# Define the paths
csv_path = "/content/drive/MyDrive/Violin/segmentation/bow_keypoints.csv"  # Replace with the actual path
output_csv_path = "/content/drive/MyDrive/Violin/segmentation/bow_movement_direction.csv"
plot_output_path = "/content/drive/MyDrive/Violin/segmentation/trajectory_plot_with_pca_smoothed.png"

# Call the function to process the movement
process_movement(csv_path, output_csv_path, plot_output_path)

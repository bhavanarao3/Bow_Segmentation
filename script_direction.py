import os
import matplotlib.pyplot as plt
from src.bow_direction_pca import compute_movement_directions

def plot_trajectory_with_pca(csv_path, output_csv_path, plot_output_path):
    # Compute movement directions and PCA coordinates
    pca_coords, smoothed_directions = compute_movement_directions(csv_path, output_csv_path)

    # Visualize the trajectory with PCA applied
    plt.figure(figsize=(1280 / 100, 720 / 100))  # Set figure size to 1280x720 pixels

    # Mark the first and last points
    plt.scatter(pca_coords[0, 0], pca_coords[0, 1], color='red', s=150, label='Start Point')
    plt.scatter(pca_coords[-1, 0], pca_coords[-1, 1], color='red', s=150, label='End Point')

    # Plot the transformed trajectory
    for i in range(len(smoothed_directions)):
        start_x, start_y = pca_coords[i, 0], pca_coords[i, 1]
        end_x, end_y = pca_coords[i + 1, 0], pca_coords[i + 1, 1]

        # Assign colors based on smoothed direction
        if smoothed_directions[i] == "Up":
            color = 'green'
        elif smoothed_directions[i] == "Down":
            color = 'red'
        else:
            color = 'yellow'

        # Plot the segment with color-coded direction
        plt.plot([start_x, end_x], [start_y, end_y], color=color, linewidth=2)

    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinate systems

    plt.title('Trajectory of Keypoint 1 (PCA Transformed with Smoothed Directions)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.legend()

    # Save the plot to a file
    plt.savefig(plot_output_path, dpi=100, bbox_inches='tight')
    plt.show()

    print(f"Trajectory plot with PCA and smoothed directions saved to {plot_output_path}")

def process_folder(input_folder, output_folder):
    # Iterate over all CSV files in the folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            csv_path = os.path.join(input_folder, file_name)
            output_csv_path = os.path.join(output_folder, f"{file_name.replace('.csv', '_direction.csv')}")
            plot_output_path = os.path.join(output_folder, f"{file_name.replace('.csv', '_trajectory_plot.png')}")

            print(f"Processing {file_name}...")
            plot_trajectory_with_pca(csv_path, output_csv_path, plot_output_path)

# Example usage
if __name__ == "__main__":
    input_folder = "result/cello/output_csv"  # Folder containing CSV files
    output_folder = "result/cello/bow_movement"  # Folder to save results

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all CSV files in the input folder
    process_folder(input_folder, output_folder)

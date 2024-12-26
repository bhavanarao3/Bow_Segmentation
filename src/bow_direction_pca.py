# src/bow_movement_pca.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt

def process_movement(csv_path, output_csv_path, plot_output_path):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Filter data for keypoint 1
    keypoint_1_df = df[df['Keypoint Index'] == 0]

    # Extract X, Y coordinates and frame numbers
    frames = keypoint_1_df['Frame'].values
    x_coords = keypoint_1_df['X'].values
    y_coords = keypoint_1_df['Y'].values

    # Combine X and Y into a single array for PCA
    coordinates = np.column_stack((x_coords, y_coords))

    # Apply PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(coordinates)

    # Extract the principal component directions
    principal_directions = pca.components_

    # Project data onto the first principal component
    projected_data = pca_coords[:, 0]

    # Determine movement direction along the principal axis
    directions = []
    for i in range(1, len(projected_data)):
        if projected_data[i] < projected_data[i - 1]:
            directions.append("Up")
        elif projected_data[i] > projected_data[i - 1]:
            directions.append("Down")
        else:
            directions.append("Stationary")

    # Apply sliding window smoothing
    window_size = 5  # Number of points in each window
    smoothed_directions = []

    for i in range(len(directions)):
        # Determine the window bounds
        start = max(0, i - window_size // 2)
        end = min(len(directions), i + window_size // 2 + 1)

        # Get the majority direction in the window
        window_directions = directions[start:end]
        most_common_direction = Counter(window_directions).most_common(1)[0][0]
        smoothed_directions.append(most_common_direction)

    # Save smoothed directions into a DataFrame
    movement_data = {
        "Frame Start": frames[:-1],
        "Frame End": frames[1:],
        "Direction": smoothed_directions
    }
    movement_df = pd.DataFrame(movement_data)

    # Save the DataFrame to a CSV file
    movement_df.to_csv(output_csv_path, index=False)
    print(f"Smoothed movement directions with PCA saved to {output_csv_path}")

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

# src/bow_movement_pca.py

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

def compute_movement_directions(csv_path, output_csv_path, window_size=5):
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Extract X, Y coordinates and frame numbers
    frames = df['Frame'].values
    x_coords = df['X'].values
    y_coords = df['Y'].values

    # Combine X and Y into a single array for PCA
    coordinates = np.column_stack((x_coords, y_coords))

    # Apply PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(coordinates)

    # Extract the principal component directions
    principal_directions = pca.components_

    # Determine which principal component has the highest variance (X or Y axis)
    if np.abs(principal_directions[0, 0]) > np.abs(principal_directions[0, 1]):
        axis = 'X'
        print('X')
    else:
        axis = 'Y'
        print('Y')

    # Project data onto the first principal component
    projected_data = pca_coords[:, 0]

    # Determine movement direction along the principal axis
    directions = []
    for i in range(1, len(projected_data)):
        if axis == 'X':  
            if projected_data[i] < projected_data[i - 1]:
                directions.append("Down")
            elif projected_data[i] > projected_data[i - 1]:
                directions.append("Up")
            else:
                directions.append("Stationary")
        elif axis == 'Y':  
            if projected_data[i] < projected_data[i - 1]:
                directions.append("Up")
            elif projected_data[i] > projected_data[i - 1]:
                directions.append("Down")
            else:
                directions.append("Stationary")

    # Apply sliding window smoothing
    smoothed_directions = []
    for i in range(len(directions)):
        start = max(0, i - window_size // 2)
        end = min(len(directions), i + window_size // 2 + 1)
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
    return pca_coords, smoothed_directions

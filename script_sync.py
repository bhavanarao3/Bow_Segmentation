import os
from src.frame_sync import annotate_frames

def process_folders(events_csv_folder, frames_csv_folder, output_folder):
    # Iterate through events CSV files in events_csv_folder
    for event_file in os.listdir(events_csv_folder):
        if event_file.endswith("_inferred_notes_with_silence.csv"):
            # Extract the base name (e.g., cello01)
            base_name = event_file.split('_')[0]  # This will give "cello01", "cello02", etc.
            
            # Match corresponding frames CSV file
            frames_csv_file = f"cello_{base_name.split('cello')[1]}_keypoints_direction.csv"
            frames_csv_path = os.path.join(frames_csv_folder, frames_csv_file)
            
            if os.path.exists(frames_csv_path):
                events_csv_path = os.path.join(events_csv_folder, event_file)
                output_csv_path = os.path.join(output_folder, f"{base_name}_annotated.csv")
                
                # Call the function to process and annotate frames
                print(f"Processing {base_name}...")
                annotate_frames(events_csv_path, frames_csv_path, output_csv_path)
            else:
                print(f"Frames CSV file for {base_name} not found. Skipping...")
                

# Example usage
if __name__ == "__main__":
    events_csv_folder = 'result/cello/output_pitch'
    frames_csv_folder = 'result/cello/bow_movement'
    output_folder = 'result/cello/annotated_output'  # Folder to save results

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process the folders
    process_folders(events_csv_folder, frames_csv_folder, output_folder)

import os
import json
from src.pitch_detection import pitch_detect_crepe, infer_note_positions_with_silence, save_inferred_notes_to_csv, plot_note_positions_with_silence

# Directory for saving output
output_dir = 'result/cello/output_pitch'
os.makedirs(output_dir, exist_ok=True)

def process_audio_file(audio_path, proj_name, output_dir, frame_start, frame_end, total_time):
    """Processes a single audio file for pitch detection and note inference."""
    # Step 1: Perform pitch detection with CREPE
    pitch_results = pitch_detect_crepe('torch', proj_name, frame_start=frame_start, 
                                       frame_end=frame_end, total_time=total_time, instrument='cello', audio_path=audio_path, 
                                       output_dir=output_dir)

    # Step 2: Infer note positions with silence detection
    inferred_notes = infer_note_positions_with_silence(pitch_results, audio_path)

    # Step 3: Save inferred notes to CSV
    save_inferred_notes_to_csv(inferred_notes, proj_name, output_dir)

    # Step 4: Plot and save note positions with silence events
    plot_note_positions_with_silence(inferred_notes, proj_name, output_dir)

def main(input_folder, json_folder, output_dir):
    """Processes all .wav files in the specified folder using information from JSON files."""
    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        # Process only .wav files
        if file_name.lower().endswith('.wav'):
            audio_path = os.path.join(input_folder, file_name)
            
            # Use the file name (without extension) as the project name
            proj_name = os.path.splitext(file_name)[0]

            # Locate the corresponding JSON file
            json_file_name = f"{proj_name}_summary.json"
            json_path = os.path.join(json_folder, json_file_name)

            if os.path.exists(json_path):
                # Load JSON data
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                # Extract frame and time information
                frame_start = json_data.get("StartFrame", 0)
                frame_end = json_data.get("EndFrame", 0)
                total_time = json_data.get("Duration", 0)

                print(f"Processing: {file_name} with project name: {proj_name}")
                print(f"Using JSON info - StartFrame: {frame_start}, EndFrame: {frame_end}, Duration: {total_time}")

                # Process the audio file with extracted values
                process_audio_file(audio_path, proj_name, output_dir, frame_start, frame_end, total_time)
            else:
                print(f"JSON file not found for: {file_name}. Skipping.")
        else:
            print(f"Skipping non-audio file: {file_name}")

# Specify the folder containing .wav files and JSON files
input_folder = 'data/cello/input_audio'  # Folder with input audio files
json_folder = 'data/cello/input_json'  # Folder with JSON files

# Run the main function
main(input_folder, json_folder, output_dir)

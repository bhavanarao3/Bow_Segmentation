import os
from src.pitch_detection import pitch_detect_crepe, infer_note_positions_with_silence, save_inferred_notes_to_csv, plot_note_positions_with_silence

# Directory for saving output
output_dir = 'result/pitch/'
os.makedirs(output_dir, exist_ok=True)

# Main code to run the entire process
def process_audio_file(audio_path, proj_name, output_dir):
    """Processes a single audio file for pitch detection and note inference."""
    # Step 1: Perform pitch detection with CREPE
    pitch_results = pitch_detect_crepe('torch', proj_name, instrument='cello', audio_path=audio_path, output_dir=output_dir)

    # Step 2: Infer note positions with silence detection
    inferred_notes = infer_note_positions_with_silence(pitch_results)

    # Step 3: Save inferred notes to CSV
    save_inferred_notes_to_csv(inferred_notes, proj_name, output_dir)

    # Step 4: Plot and save note positions with silence events
    plot_note_positions_with_silence(inferred_notes, proj_name, output_dir)

def main(input_folder, output_dir):
    """Processes all .wav files in the specified folder."""
    # Iterate through all files in the input folder
    for file_name in os.listdir(input_folder):
        # Process only .wav files
        if file_name.lower().endswith('.wav'):
            audio_path = os.path.join(input_folder, file_name)
            
            # Use the file name (without extension) as the project name
            proj_name = os.path.splitext(file_name)[0]
            
            print(f"Processing: {file_name} with project name: {proj_name}")
            
            # Process the audio file
            process_audio_file(audio_path, proj_name, output_dir)
        else:
            print(f"Skipping non-audio file: {file_name}")

# Specify the folder containing .wav files
input_folder = 'data/input_audio/'  # Folder with input audio files

# Run the main function
main(input_folder, output_dir)

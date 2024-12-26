# main.py
import os
from src.pitch_detection import pitch_detect_crepe, infer_note_positions_with_silence, save_inferred_notes_to_csv, plot_note_positions_with_silence

# Directory for saving output
output_dir = '/content/drive/MyDrive/Violin/segmentation'
os.makedirs(output_dir, exist_ok=True)

# Main code to run the entire process
def main(audio_path, proj_name, output_dir):
    # Step 1: Perform pitch detection with CREPE
    pitch_results = pitch_detect_crepe('torch', proj_name, instrument='cello', audio_path=audio_path, output_dir=output_dir)

    # Step 2: Infer note positions with silence detection
    inferred_notes = infer_note_positions_with_silence(pitch_results)

    # Step 3: Save inferred notes to CSV
    save_inferred_notes_to_csv(inferred_notes, proj_name, output_dir)

    # Step 4: Plot and save note positions with silence events
    plot_note_positions_with_silence(inferred_notes, proj_name, output_dir)

# Specify the path of the .wav file and project name
audio_path = '/content/drive/MyDrive/Violin/cello01.wav'
project_name = 'pitch_detection_project'

# Run the main function
main(audio_path, project_name, output_dir)

# src/pitch_detection.py
import os
import numpy as np
import pandas as pd
import librosa
import crepe
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Utility function to map frequency (Hz) to musical note
def hz_to_note_name(hz):
    if hz is None or np.isnan(hz):
        return None
    note_num = int(round(12 * np.log2(hz / 440.0))) + 69  # Convert to MIDI number
    return librosa.midi_to_note(note_num)

# Function to draw the fundamental pitch curve
def draw_fundamental_curve(time, frequency, confidence, proj, algo, output_dir):
    plt.figure(figsize=(14, 5))
    plt.plot(time, frequency, label='Frequency (pitch)', color='blue', linewidth=1.5)
    plt.fill_between(time, 0, confidence, color='gray', alpha=0.5, label='Confidence')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Pitch Curve - {algo.upper()}')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save plot as .jpg file
    output_path = f"{output_dir}/{proj}_pitch_curve_{algo}.jpg"
    plt.savefig(output_path)
    plt.close()
    print(f"Pitch curve saved as: {output_path}")

# Pitch detection with CREPE model (either torch or tensorflow backend)
def pitch_detect_crepe(crepe_backend, proj, frame_start, frame_end, total_time, instrument='cello', audio_path='wavs/background.wav', output_dir='output'):
    if crepe_backend == 'torch':
        import torchcrepe
        import torch
        y, sr = librosa.load(audio_path, mono=True)
        audio_1channel = torch.tensor(y).reshape(1, -1)
        sample_num = audio_1channel.shape[1]

        # Define frequency range based on the instrument
        if instrument == 'cello':
            min_freq, max_freq = 65, 1047  # Approx range for cello
        else:
            min_freq, max_freq = 196, 3136  # Approx range for violin

        frequency, confidence = torchcrepe.predict(
            audio_1channel,
            sr,
            hop_length=int(sr / 30.),
            return_periodicity=True,
            model='full',
            fmin=min_freq,
            fmax=max_freq,
            batch_size=2048,
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # Reshape results for further processing
        frequency = frequency.detach().cpu().numpy().reshape(-1,)
        confidence = confidence.detach().cpu().numpy().reshape(-1,)
        time = np.linspace(0, sample_num / sr, len(frequency))

    elif crepe_backend == 'tensorflow':
        sr, y = wavfile.read(audio_path)
        time, frequency, confidence, activation = crepe.predict(
            y, sr, viterbi=True, step_size=100 / 3, model_capacity='full', center=True
        )

    else:
        print('Please specify crepe_backend as either "torch" or "tensorflow"')
        return None

    # Map time to frame numbers
    frames = frame_start + (time / total_time) * (frame_end - frame_start)


    # Plot and save the pitch curve
    draw_fundamental_curve(time, frequency, confidence, proj, 'crepe', output_dir)

    # Save the pitch results as CSV with the updated time stamps
    pitch_results = np.stack((time, frequency, confidence, frames), axis=1)
    output_csv_path = f"{output_dir}/{proj}_pitch_frequencies.csv"
    pd.DataFrame(pitch_results, columns=["Time (s)", "Frequency (Hz)", "Confidence", "Frame"]).to_csv(output_csv_path, index=False)
    print(f"Pitch frequencies saved as: {output_csv_path}")

    return pitch_results

# Inference function to estimate note positions with silence detection
def infer_note_positions_with_silence(pitch_results):
    inferred_notes = []
    previous_note = None

    for time, freq, prob, frame in pitch_results:
        # Check for silence (no bow contact)
        if prob < 0.6:
            inferred_notes.append((frame, time, None, "silence"))
        else:
            # Convert frequency to musical note if probability of voicing is high enough
            current_note = hz_to_note_name(freq)

            # Detect changes in notes to infer position changes
            if current_note != previous_note:
                inferred_notes.append((frame, time, current_note, "change"))
                previous_note = current_note
            else:
                inferred_notes.append((frame, time, current_note, "sustain"))

    return inferred_notes

# Plot function for note positions, changes, and no bow contact events
def plot_note_positions_with_silence(inferred_notes, proj, output_dir):
    times = [item[1] for item in inferred_notes]
    notes = [item[2] if item[2] is not None else 'Rest' for item in inferred_notes]
    events = [item[3] for item in inferred_notes]

    plt.figure(figsize=(14, 5))

    # Map notes to y-axis positions for plotting
    unique_notes = sorted(set(notes))  # Get unique notes including 'Rest'
    note_to_y = {note: i for i, note in enumerate(unique_notes)}
    y_values = [note_to_y[note] for note in notes]

    # Plot the note positions with markers for changes, sustains, and no bow contact
    for i, event in enumerate(events):
        if event == "change":
            plt.plot(times[i], y_values[i], marker='o', markersize=8, color='green', label="Change" if i == 0 else "")
        elif event == "sustain":
            plt.plot(times[i], y_values[i], marker='x', markersize=5, color='red', label="Sustain" if i == 0 else "")
        elif event == "silence":
            plt.plot(times[i], y_values[i], marker='s', markersize=8, color='gray', label="No Bow Contact" if i == 0 else "")

    plt.yticks(ticks=list(range(len(unique_notes))), labels=unique_notes)
    plt.xlabel("Time (s)")
    plt.ylabel("Note")
    plt.title("Inferred Note-Playing Positions with Silence Events")
    plt.legend(loc="upper right")
    plt.grid(True)

    # Save plot as .jpg in specified path
    output_path = os.path.join(output_dir, f"{proj}_note_positions_with_silence.jpg")
    plt.savefig(output_path)
    plt.close()
    print(f"Note positions plot saved as: {output_path}")

# Save inferred notes to CSV
def save_inferred_notes_to_csv(inferred_notes, proj, output_dir):
    df_inferred = pd.DataFrame(inferred_notes, columns=["Frame", "Time (s)", "Note", "Event"])
    output_csv_path = os.path.join(output_dir, f"{proj}_inferred_notes_with_silence.csv")
    df_inferred.to_csv(output_csv_path, index=False)
    print(f"Inferred notes saved as: {output_csv_path}")

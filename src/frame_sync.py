import pandas as pd

def annotate_frames(events_csv_path, frames_csv_path, output_csv_path):
    """
    Annotates frames based on events and frame range data, handling silences and directions.

    Parameters:
    - events_csv_path (str): Path to the events CSV file.
    - frames_csv_path (str): Path to the frames CSV file.
    - output_csv_path (str): Path to save the annotated CSV file.

    Returns:
    - None
    """
    # Read the CSV files
    events_df = pd.read_csv(events_csv_path)  # File with Time (s), Note, Event, Frame
    frames_df = pd.read_csv(frames_csv_path)  # File with Frame Start, Frame End, Direction

    # Add 300 to all numbers in Frame Start and Frame End columns
    frames_df['Frame Start'] += 300
    frames_df['Frame End'] += 300

    # Add a 'SilenceGroup' column to identify consecutive silence sequences
    events_df['IsSilence'] = events_df['Event'].str.lower() == 'silence'

    # Use a helper column to identify consecutive silence sequences
    events_df['SilenceGroup'] = (events_df['IsSilence'] != events_df['IsSilence'].shift()).cumsum()

    # Find groups with 6 or more consecutive silences
    silence_groups = events_df.groupby('SilenceGroup')['IsSilence'].sum()
    valid_silence_groups = silence_groups[silence_groups >= 6].index

    # Filter and annotate based only on `frames_df`
    annotations = []
    for _, frame_row in frames_df.iterrows():
        frame_start, frame_end, direction = frame_row['Frame Start'], frame_row['Frame End'], frame_row['Direction']
        
        # Check if any event falls within the frame range
        matching_events = events_df[
            (events_df['Frame'] >= frame_start) & (events_df['Frame'] <= frame_end)
        ]
        
        for _, event_row in matching_events.iterrows():
            frame = event_row['Frame']
            event = event_row['Event']
            
            # Annotate as silence if it belongs to a valid silence group
            if event_row['SilenceGroup'] in valid_silence_groups:
                annotation = 'silence'
            else:
                annotation = direction  # Use the direction from `frames_df`
            
            annotations.append({
                'Frame': frame,
                'Time (s)': event_row['Time (s)'],
                'Note': event_row['Note'],
                'Event': event,
                'Annotation': annotation
            })

    # Convert results to DataFrame
    annotated_frames_df = pd.DataFrame(annotations)
    annotated_frames_df['Frame'] = annotated_frames_df['Frame'] - 300  # Subtract 300 from Frame values
    annotated_frames_df['Frame'] = annotated_frames_df['Frame'].astype(int)  # Ensure Frame is an integer

    # Save the annotated DataFrame to CSV
    annotated_frames_df.to_csv(output_csv_path, index=False)
    print(f"Annotated frames saved to: {output_csv_path}")

import cv2
import pandas as pd

def annotate_video(base_video_path, movement_csv_path, output_video_path):
    """
    Annotates the video with movement directions based on the provided CSV file.

    Parameters:
    - base_video_path (str): Path to the input video.
    - movement_csv_path (str): Path to the CSV file containing movement annotations.
    - output_video_path (str): Path to save the annotated video.

    Returns:
    - None
    """
    # Load movement direction data
    movement_df = pd.read_csv(movement_csv_path)

    # Open base video
    base_video = cv2.VideoCapture(base_video_path)
    fps = int(base_video.get(cv2.CAP_PROP_FPS))
    frame_width = int(base_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(base_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(base_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Arrow position on the mid-right part of the frame
    arrow_start = (frame_width - 150, frame_height // 2)
    arrow_length = 300

    # Process each frame
    frame_idx = 0
    while frame_idx < total_frames:
        ret, base_frame = base_video.read()
        if not ret:
            break

        # Get the movement direction for the current frame
        movement = movement_df[movement_df['Frame'] == frame_idx]
        if not movement.empty:
            direction = movement.iloc[0]['Annotation']
            if direction == "Up":
                color = (0, 255, 0)  # Green for upward movement
                arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
                # Draw the arrow for upward movement
                cv2.arrowedLine(
                    base_frame,
                    arrow_start,
                    arrow_end,
                    color,
                    thickness=3,
                    tipLength=0.4
                )
            elif direction == "Down":
                color = (0, 0, 255)  # Red for downward movement
                arrow_end = (arrow_start[0], arrow_start[1] + arrow_length)
                # Draw the arrow for downward movement
                cv2.arrowedLine(
                    base_frame,
                    arrow_start,
                    arrow_end,
                    color,
                    thickness=3,
                    tipLength=0.4
                )
            elif direction == "silence":
                color = (255, 255, 0)  # Yellow for silence
                # Draw a horizontal line for silence
                cv2.line(
                    base_frame,
                    (arrow_start[0], arrow_start[1] - 150),  # Start of the line
                    (arrow_start[0], arrow_start[1] + 150),  # End of the line
                    color,
                    thickness=3
                )

        # Write the annotated frame to the output video
        out.write(base_frame)
        frame_idx += 1

    # Release resources
    base_video.release()
    out.release()

    print(f"Combined video with direction arrows saved to: {output_video_path}")

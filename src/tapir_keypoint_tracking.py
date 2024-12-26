import functools
import jax
import mediapy as media
import numpy as np
import csv
import cv2  # Import OpenCV for drawing
from tapnet.models import tapir_model
from tapnet.utils import transforms, model_utils

MODEL_TYPE = 'tapir'  # 'tapir' or 'bootstapir'

class TapirKeypointTracking:
    def __init__(self, checkpoint_path):
        # Load the TAPIR model
        ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']

        kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
        if MODEL_TYPE == 'bootstapir':
            kwargs.update(dict(
                pyramid_level=1,
                extra_convs=True,
                softmax_temperature=10.0
            ))

        self.tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

    def inference(self, frames, query_points):
        """Inference on one video."""
        frames = model_utils.preprocess_frames(frames)
        query_points = query_points.astype(np.float32)
        frames, query_points = frames[None], query_points[None]  # Add batch dimension
        outputs = self.tapir(video=frames, query_points=query_points, is_training=False, query_chunk_size=32)
        tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
        return tracks[0], visibles[0]

    def track_keypoints(self, video_path, keypoints, output_video_path, keypoints_file_path):
        # Load the video
        video = media.read_video(video_path)

        # Get original video dimensions
        orig_height, orig_width = video.shape[1:3]
        print(f"Original video dimensions: {orig_height}x{orig_width}")

        # Downsample the video to reduce computational load
        downsampled_video = media.resize_video(video, (256, 256))

        # Perform inference
        inference = jax.jit(self.inference)
        tracks, visibles = inference(downsampled_video, keypoints)
        tracks = np.array(tracks)
        visibles = np.array(visibles)

        # Swap the first two axes (from (10, 302, 2) to (302, 10, 2))
        tracks = np.swapaxes(tracks, 0, 1)
        visibles = np.swapaxes(visibles, 0, 1)

        # Convert tracked points back to the original video dimensions
        tracks_orig_dims = transforms.convert_grid_coordinates(tracks, (256, 256), (orig_width, orig_height))

        # Prepare to draw the keypoints on the video frames
        output_frames = video.copy()  # Make a copy of the video frames for modification

        # Draw the keypoints on the frames using OpenCV
        for frame_idx in range(tracks_orig_dims.shape[0]):
            for kp_idx in range(tracks_orig_dims.shape[1]):
                if visibles[frame_idx, kp_idx]:
                    x, y = tracks_orig_dims[frame_idx, kp_idx]
                    # Draw a small circle on the keypoint (OpenCV)
                    output_frames[frame_idx] = cv2.circle(output_frames[frame_idx], (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)

        # Save the modified video with keypoints plotted
        media.write_video(output_video_path, output_frames, fps=10)
        print(f"Output video saved as: {output_video_path}")

        # Save visible keypoints to a CSV file
        with open(keypoints_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'Keypoint Index', 'X', 'Y'])

            num_frames, num_keypoints = tracks_orig_dims.shape[0], tracks_orig_dims.shape[1]
            for frame_idx in range(num_frames):
                for kp_idx in range(num_keypoints):
                    if visibles[frame_idx, kp_idx]:
                        x, y = tracks_orig_dims[frame_idx, kp_idx]
                        writer.writerow([frame_idx, kp_idx, x, y])

        print(f"Visible keypoints saved to: {keypoints_file_path}")

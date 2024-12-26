# Bow_Segmentation

This project provides a pipeline for analyzing and visualizing bow movements in string instrument performance videos. It involves tracking keypoints, analyzing pitch, determining movement direction, synchronizing data, and visualizing results in an annotated video.

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd bow_segmentation
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Workflow

### Track Keypoints
The first step involves tracking keypoints in the video.

#### TAPIR
Run the following command:

```bash
python script_keypoint.py
```
This script generates a file with the tracked keypoints and coordinates for subsequent steps.

#### METRABS

```bash
python script_metrabs.py
```
Note: For Metrabs keypoint tracking, ensure that the video is upright and not rotated.

### Pitch Analysis
Analyze the pitch in the audio data to identify silent frames.

Run the following command:

```bash
python script_pitch.py
```
This script outputs a list of frames marked as silent based on the pitch analysis.

### Determine Movement Direction
Use the keypoint tracking data and apply PCA to calculate the direction of bow movement.

Run the following command:

```bash
python script_direction.py
```
This script outputs the direction for each frame (Up or Down).

### Sync Frames
Synchronize the movement directions and silent frames to generate final annotations for each frame.

Run the following command:

```bash
python script_sync.py
```
The result will be an annotated list of frames with labels (Up, Down, or Silence).

### Visualize Results
Visualize the annotated data by creating a video with arrows and markers for movement direction.

Run the following command:

```bash
python script_visualization.py
```
This will produce an annotated video saved in the result directory.

## Output
- Keypoint Tracking: Tracked keypoints and coordinates.
- Pitch Analysis: Identified silent frames.
- Movement Direction: Frames annotated with directions (Up or Down).
- Synchronized Data: Final frame-level annotations (Up, Down, Silence`).
- Visualization: A video with annotations overlayed to show movement directions and silence.


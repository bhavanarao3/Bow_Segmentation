from moviepy.video.io.VideoFileClip import VideoFileClip

def convert_mp4_to_wav(mp4_path, wav_path):
    """
    Converts an MP4 video file to a WAV audio file.

    Args:
        mp4_path (str): Path to the input MP4 file.
        wav_path (str): Path to save the output WAV file.
    """
    try:
        # Load the video file
        video = VideoFileClip(mp4_path)
        
        # Extract audio
        audio = video.audio
        
        # Write audio to .wav file
        audio.write_audiofile(wav_path)
        
        print(f"Conversion successful! Saved to: {wav_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
mp4_path = '/content/drive/MyDrive/Violin/youtube/1.mp4'  # Replace with the path to your .mp4 file
wav_path = '/content/drive/MyDrive/Violin/segmentation/1.wav'  # Replace with the desired output path for the .wav file

convert_mp4_to_wav(mp4_path, wav_path)

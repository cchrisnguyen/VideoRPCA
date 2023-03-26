import numpy as np
import cv2

def make_video(frames, height=320, width=256, filename='output.mp4', fps=30):
    """
    The make_video function takes a 2D array of frames and writes them to a video file.

    Args:
        frames (numpy.ndarray): A 2D array of frames to be written to a video file. 
            The array has dimensions n_frames x (height x width), where n_frames is the number of frames in the video.
        height (int): the height dimension of each frame. 
        width (int): the width dimension of each frame.
        filename (str): the output file's name.
        fps (int): frame per second of the output video.

    Returns:
        None.
    """
    n_frames = frames.shape[0]
    # Reshape back to video 
    video = np.reshape(frames, (n_frames, height, width))
    # Rotate each image by 90 degrees clockwise
    video = np.rot90(video, k=-1, axes=(1, 2))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, fps, (height, width), isColor=False)
    for i in range(n_frames):
        frame = video[i]
        frame = np.uint8(frame)
        writer.write(frame)
    writer.release()

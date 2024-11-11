import cv2
import os
import numpy as np
import time

from opticalflow_our_method import (
    split_video_into_frames,
    enhance_image,
    horn_schunk,
    get_magnitude
)


def compare_videos(video1_path, video2_path, output_dir="comparison_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Split both videos into frames
    print("Extracting frames from the first video...")
    frames1, fps1 = split_video_into_frames(video1_path)
    
    print("Extracting frames from the second video...")
    frames2, fps2 = split_video_into_frames(video2_path)

    # Ensure both videos have the same number of frames for comparison
    print(len(frames1), "video 1 loaded")
    print(len(frames2), "video 2 loaded")
    
    # Prepare metrics
    magnitude_diffs = []
    mse_values = []
    angle_diffs = []
    processing_times_video1 = []
    processing_times_video2 = []

    # Process frame pairs
    for i, (frame1, frame2) in enumerate(zip(frames1, frames2)):
        print(f"Processing frame pair {i+1}...")

        # Calculate start time
        start_time_1 = time.time()
        u1, v1 = horn_schunk(frame1, frame1)  # Optical flow for video 1
        processing_time_1 = time.time() - start_time_1
        processing_times_video1.append(processing_time_1)

        start_time_2 = time.time()
        u2, v2 = horn_schunk(frame2, frame2)  # Optical flow for video 2
        processing_time_2 = time.time() - start_time_2
        processing_times_video2.append(processing_time_2)

        # Calculate magnitudes and difference
        mag1 = get_magnitude(u1, v1)
        mag2 = get_magnitude(u2, v2)
        magnitude_diffs.append(abs(mag1 - mag2))

        # Calculate pixel-wise Mean Squared Error (MSE) between frames
        img1 = cv2.imread(frame1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(frame2, cv2.IMREAD_GRAYSCALE)
        mse = np.mean((img1 - img2) ** 2)
        mse_values.append(mse)

        # Calculate flow vector angle differences
        angles_video1 = np.arctan2(v1, u1)  # Flow vector angles for video 1
        angles_video2 = np.arctan2(v2, u2)  # Flow vector angles for video 2
        angle_diff = np.mean(np.abs(angles_video1 - angles_video2))
        angle_diffs.append(angle_diff)

        print(f"Frame {i+1} Analysis:")
        print(f"  Magnitude Difference: {magnitude_diffs[-1]}")
        print(f"  MSE: {mse}")
        print(f"  Average Angle Difference: {angle_diff}")
        print(f"  Processing Time Video 1: {processing_time_1} seconds")
        print(f"  Processing Time Video 2: {processing_time_2} seconds")

    # Calculate overall summary metrics
    avg_magnitude_diff = np.mean(magnitude_diffs)
    avg_mse = np.mean(mse_values)
    avg_angle_diff = np.mean(angle_diffs)
    avg_processing_time_video1 = np.mean(processing_times_video1)
    avg_processing_time_video2 = np.mean(processing_times_video2)

    # Display summary results
    print("\n=== Summary of Video Comparison ===")
    print(f"Average Magnitude Difference: {avg_magnitude_diff}")
    print(f"Average MSE between frames: {avg_mse}")
    print(f"Average Angle Difference between flow vectors: {avg_angle_diff}")
    print(f"Total Processing Time for Video 1: {sum(processing_times_video1)} seconds")
    print(f"Total Processing Time for Video 2: {sum(processing_times_video2)} seconds")
    print(f"Average Processing Time per Frame for Video 1: {avg_processing_time_video1} seconds")
    print(f"Average Processing Time per Frame for Video 2: {avg_processing_time_video2} seconds")

    if avg_magnitude_diff < 0.1 and avg_mse < 100 and avg_angle_diff < 0.1:
        print("The two videos are highly similar in terms of optical flow and pixel intensity.")
    else:
        print("The two videos show noticeable differences in optical flow and/or pixel intensity.")

    # Clean up frame images
    for frame in frames1 + frames2:
        os.remove(frame)
    print("\nFrame images cleaned up.")

if __name__ == "__main__":
    video1 = input("Enter path to first video: ")
    video2 = input("Enter path to second video: ")
    compare_videos(video1, video2)

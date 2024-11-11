import numpy as np
import cv2

def calculate_optical_flow(prev_frame, curr_frame, window_size, min_quality=0.01):
    max_corners = 10000
    min_distance = 0.1
    feature_params = dict(maxCorners=max_corners, qualityLevel=min_quality, minDistance=min_distance)
    points = cv2.goodFeaturesToTrack(prev_frame, **feature_params)

    half_window_size = window_size // 2

    prev_frame = prev_frame / 255.0
    curr_frame = curr_frame / 255.0

    # image gradients
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])
    Ix = cv2.filter2D(prev_frame, -1, kernel_x)
    Iy = cv2.filter2D(prev_frame, -1, kernel_y)
    It = cv2.filter2D(curr_frame, -1, kernel_t) - cv2.filter2D(prev_frame, -1, kernel_t)

    # Make flow_vectors a two-channel array to store (u, v) for each point
    flow_vectors = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 2), dtype=np.float32)

    for point in points:
        x, y = point.ravel()
        x, y = int(x), int(y)

        window_x = Ix[y - half_window_size:y + half_window_size + 1, x - half_window_size:x + half_window_size + 1]
        window_y = Iy[y - half_window_size:y + half_window_size + 1, x - half_window_size:x + half_window_size + 1]
        window_t = It[y - half_window_size:y + half_window_size + 1, x - half_window_size:x + half_window_size + 1]

        A = np.vstack((window_x.flatten(), window_y.flatten())).T
        b = -window_t.flatten()

        # Solve the least-squares
        u, v = np.linalg.lstsq(A, b, rcond=None)[0]

        if is_within_bounds((x + int(u), y + int(v)), prev_frame.shape):
            flow_vectors[y, x] = [u, v]  # Use [u, v] to assign values to each channel

    return flow_vectors




def is_within_bounds(point, image_shape):
    x, y = point
    height, width = image_shape
    return 0 <= x < width and 0 <= y < height

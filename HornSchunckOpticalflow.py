import cv2
import numpy as np
from scipy.ndimage import convolve as filter2


def get_derivatives(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1, x_kernel) + filter2(img2, x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx, fy, ft]

def OpticalFlowVectors(img1_path, img2_path, output_path, alpha=15, delta=40**-1):
    beforeImg = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    afterImg = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    beforeImg = beforeImg.astype(float)
    afterImg = afterImg.astype(float)

    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                           [1 / 6, 0, 1 / 6],
                           [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        if diff < delta or iter_counter > 100:
            break

    return u, v



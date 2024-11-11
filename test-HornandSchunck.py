import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve as filter2

def calculate_image_gradients(img1, img2):
    dx = np.array([[-1, 1], [-1, 1]]) * 0.25
    dy = np.array([[-1, -1], [1, 1]]) * 0.25
    dt = np.array([[1, 1], [1, 1]]) * 0.25

    Ix = filter2(img1, dx) + filter2(img2, dx)
    Iy = filter2(img1, dy) + filter2(img2, dy)
    It = filter2(img1, -dt) + filter2(img2, dt)

    return Ix, Iy, It

def horn_schunck(img1, img2, alpha, max_iterations=300, tolerance=1e-3):
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    Ix, Iy, It = calculate_image_gradients(img1, img2)

    for _ in range(max_iterations):
        u_avg = (u[1:, :] + u[:-1, :] + u[:, 1:] + u[:, :-1]) / 4
        v_avg = (v[1:, :] + v[:-1, :] + v[:, 1:] + v[:, :-1]) / 4

        p = Ix * u_avg + Iy * v_avg + It
        d = alpha**2 + Ix**2 + Iy**2

        u = u_avg - (Ix * p) / d
        v = v_avg - (Iy * p) / d

        if np.linalg.norm(u - u_avg) + np.linalg.norm(v - v_avg) < tolerance:
            break

    return u, v

def overlay_flow(img, u, v, output_path):
    plt.figure()
    plt.imshow(img, cmap='gray')

    for y in range(0, img.shape[0], 8):
        for x in range(0, img.shape[1], 8):
            dx = u[y, x] * 3
            dy = v[y, x] * 3
            plt.arrow(x, y, dx, dy, color='red', head_width=2, head_length=3)

    plt.savefig(output_path)
    plt.show()

if __name__ == '__main__':
    img1_path = input("enter first image: ")
    img2_path = input("enter second image: ")
    output_path = 'output-hornschunk-'+img1_path[:-4]+ '_' + img2_path[:-4]+'.png'

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    u, v = horn_schunck(img1, img2, alpha=15)

    overlay_flow(img1, u, v, output_path)
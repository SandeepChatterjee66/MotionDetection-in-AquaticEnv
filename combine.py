import cv2
import numpy as np

# attempt to combine hornskunk with other methods

def calculate_structure_tensor(img, rho):
    Ix, Iy = cv2.gradient(img, cv2.CV_64F)

    Jxx = cv2.GaussianBlur(Ix**2, (2*rho+1, 2*rho+1), sigmaX=rho)
    Jyy = cv2.GaussianBlur(Iy**2, (2*rho+1, 2*rho+1), sigmaX=rho)
    Jxy = cv2.GaussianBlur(Ix*Iy, (2*rho+1, 2*rho+1), sigmaX=rho)

    return np.dstack([Jxx, Jxy, Jxy, Jyy])

def combined_optical_flow(img1, img2, alpha=0.5, iterations=10, rho=1.0):
    Ix, Iy, It = cv2.gradient(img1, cv2.CV_64F)

    J = calculate_structure_tensor(img1, rho)

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    # Iterative minimization
    for _ in range(iterations):
        u_new = u.copy()
        v_new = v.copy()

        for y in range(1, img1.shape[0] - 1):
            for x in range(1, img1.shape[1] - 1):
                J_x = J[y, x, 0]
                J_y = J[y, x, 1]
                J_xy = J[y, x, 2]
                J_yy = J[y, x, 3]

                div_u = (u[y+1, x] - u[y-1, x]) / 2 + (u[y, x+1] - u[y, x-1]) / 2
                div_v = (v[y+1, x] - v[y-1, x]) / 2 + (v[y, x+1] - v[y, x-1]) / 2

                laplacian_u = u[y+1, x] + u[y-1, x] + u[y, x+1] + u[y, x-1] - 4 * u[y, x]
                laplacian_v = v[y+1, x] + v[y-1, x] + v[y, x+1] + v[y, x-1] - 4 * v[y, x]

                u_new[y, x] = u[y, x] - (J_x * div_u + J_xy * div_v + alpha * laplacian_u - Ix * It) / (J_x * J_x + 2 * J_xy * J_xy + J_yy * J_yy + alpha)
                v_new[y, x] = v[y, x] - (J_xy * div_u + J_yy * div_v + alpha * laplacian_v - Iy * It) / (J_x * J_x + 2 * J_xy * J_xy + J_yy * J_yy + alpha)

        u = u_new
        v = v_new

    return u, v

img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

u, v = combined_optical_flow(img1, img2)

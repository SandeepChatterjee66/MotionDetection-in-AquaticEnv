import time
import cv2
import numpy as np

"""Compares the performance of three optical flow methods."""

from opticalflow_baseline1_LucasKanade import calculate_optical_flow
from opticalflow_baseline2_RAFTdeeplearning import estimate_flow
from opticalflow_our_method import horn_schunk

def compare_methods(image_pairs):
    for img1_path, img2_path in image_pairs:
        print(f"Processing image pair: {img1_path}, {img2_path}")

        # Lucas-Kanade
        start_time = time.time()
        flow_lk = calculate_optical_flow(cv2.imread(img1_path), cv2.imread(img2_path), window_size=15)
        end_time = time.time()
        lk_time = end_time - start_time
        print(f"Lucas-Kanade time: {lk_time:.2f} seconds")

        # RAFT
        start_time = time.time()
        flow_raft = estimate_flow(img1_path, img2_path)
        end_time = time.time()
        raft_time = end_time - start_time
        print(f"RAFT time: {raft_time:.2f} seconds")

        # Horn-Schunck
        start_time = time.time()
        flow_hs = horn_schunk(img1_path, img2_path, "frame_pair")
        end_time = time.time()
        hs_time = end_time - start_time
        print(f"Horn-Schunck time: {hs_time:.2f} seconds")

        # Visualize the flow fields
        h, w = flow_lk.shape[:2]
        flow_image_lk = np.zeros((h, w, 3), dtype=np.uint8)
        flow_image_raft = np.zeros((h, w, 3), dtype=np.uint8)
        flow_image_hs = np.zeros((h, w, 3), dtype=np.uint8)

        # Visualize Lucas-Kanade flow
        cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY), 0.5, flow_viz.flow_to_image(flow_lk), 0.5, 0), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB, flow_image_lk)

        # Visualize RAFT flow
        cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY), 0.5, flow_viz.flow_to_image(flow_raft), 0.5, 0), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB, flow_image_raft)

        # Visualize Horn-Schunck flow
        cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY), 0.5, flow_viz.flow_to_image(flow_hs), 0.5, 0), cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB, flow_image_hs)

        # Display the flow images
        cv2.imshow("Lucas-Kanade Flow", flow_image_lk)
        cv2.imshow("RAFT Flow", flow_image_raft)
        cv2.imshow("Horn-Schunck Flow", flow_image_hs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_pairs = [
        ("t1.jpg", "t2.jpg"),
        ("t2.jpg", "t3.jpg"),
        ("t3.jpg", "t4.jpg"),
        ("t4.jpg", "t5.jpg"),
        ("t5.jpg", "t6.jpg")
    ]

    compare_methods(image_pairs)
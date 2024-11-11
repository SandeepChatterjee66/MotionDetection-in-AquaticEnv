import cv2
import numpy as np
import sys
from PIL import Image, ImageEnhance
import os

def enhance_image(input_path):
    # Check if the input file exists
    if not os.path.isfile(input_path):
        print(f"Error: The file {input_path} does not exist.")
        sys.exit(1)
        
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not open or find the image {input_path}")
        sys.exit(1)

    # Convert to RGB (OpenCV reads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use PIL to adjust brightness and contrast
    pil_img = Image.fromarray(img)

    # Enhance contrast
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.5)  # Adjust contrast level as needed

    # Enhance brightness
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.2)  # Adjust brightness level as needed

    # Convert back to OpenCV format
    img = np.array(pil_img)

    # Apply bilateral filter to emphasize the fish's edges
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert to grayscale and threshold to create a mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Apply Gaussian Blur to the background
    background_blur = cv2.GaussianBlur(img, (15, 15), 0)

    # Combine blurred background with original fish area
    img = np.where(mask[..., None] == 255, img, background_blur)

    # Convert back to BGR for saving with OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Prepare output directory and path
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"enhanced-{os.path.basename(input_path)}")

    # Save the enhanced image
    cv2.imwrite(output_path, img)
    print(f"Enhanced image saved to {output_path}")

if __name__ == "__main__":
    # Check for correct usage
    if len(sys.argv) != 2:
        print("Usage: python enhance.py <filename>")
        sys.exit(1)

    # Get the input file name and prepare the input path
    input_file = sys.argv[1]
    input_path = os.path.join("frames", input_file)

    # Run the enhancement
    enhance_image(input_path)

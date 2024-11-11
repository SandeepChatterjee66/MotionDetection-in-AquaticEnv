import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from matplotlib import pyplot as plt
from scipy.ndimage import convolve as filter2
import imageio
import sys
from PIL import Image, ImageEnhance

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
FRAME_FOLDER = 'frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['FRAME_FOLDER'] = FRAME_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)


def convolve(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def enhance_image(input_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error to find the image {input_path}")
        sys.exit(1)      
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # (50% red, 25% green, 25% blue)
    img = img.astype(np.float32)
    img[:, :, 0] *= 0.5  # Red channel
    img[:, :, 1] *= 0.25  # Green channel
    img[:, :, 2] *= 0.25  # Blue channel
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Gaussian blur kernel (5x5)
    gaussian_kernel = (1 / 256) * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    img = convolve(img, gaussian_kernel)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    img = convolve(img, sharpening_kernel)

    pil_img = Image.fromarray(img)
    contrast_enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = contrast_enhancer.enhance(1.5) 
    brightness_enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = brightness_enhancer.enhance(1.2)
    img = np.array(pil_img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    output_path = input_path 
    cv2.imwrite(output_path, img)
    print(f"Enhanced image saved to {output_path}")


def split_video_into_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Select only 40 frames from the video
    frame_interval = total_frames // 40
    frame_names = []

    for i in range(40):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(FRAME_FOLDER, f't{i+1}.jpg')
            frame_names.append(frame_name)
            cv2.imwrite(frame_name, frame)
            enhance_image(frame_name)
    cap.release()
    return frame_names, fps




def gradients(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1, x_kernel) + filter2(img2, x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx, fy, ft]




def horn_schunk(img1_path, img2_path, frame_pair, alpha=15, delta=40**-1):
    beforeImg = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    afterImg = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    beforeImg = beforeImg.astype(float)
    afterImg = afterImg.astype(float)


    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = gradients(beforeImg, afterImg)
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
        if diff < delta or iter_counter > 300:
            break

    output_path = os.path.join(OUTPUT_FOLDER, f'{frame_pair}_output.png')
    flow_overlay(u, v, beforeImg, output_path)

    return output_path





def get_magnitude(u, v):
    scale=3
    dx = u * scale
    dy = v * scale
    magnitude = np.sqrt(dx**2 + dy**2)
    mag_avg = np.mean(magnitude)
    return mag_avg

def flow_overlay(u, v, beforeImg, output_path):
    scale = 3
    ax = plt.figure().gca()
    ax.imshow(beforeImg, cmap='gray')

    magnitudeAvg = magnitude(u, v)

    for i in range(0, u.shape[0], 8):
        for j in range(0, u.shape[1], 8):
            dy = v[i, j] * scale
            dx = u[i, j] * scale
            magnitude = (dx**2 + dy**2)**0.5
            if magnitude > magnitudeAvg:
                ax.arrow(j, i, dx, dy, color='red')
    plt.draw()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    

def create_gif_from_frames(frame_names, gif_output_path, fps):
    frames = []
    min_height = min(imageio.imread(frame_name).shape[0] for frame_name in frame_names)
    min_width = min(imageio.imread(frame_name).shape[1] for frame_name in frame_names)
    for frame_name in frame_names:
        frame = imageio.imread(frame_name)
        resized_frame = cv2.resize(frame, (min_width, min_height))
        frames.append(resized_frame)
    imageio.mimsave(gif_output_path, frames, duration=1/fps)







@app.route('/')
def index():
    return render_template('index.html')



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)
        frame_names, fps = split_video_into_frames(video_path)

        result_frames = []
        for i in range(39):  # We have 40 frames, so 39 pairings
            img1_path = frame_names[i]
            img2_path = frame_names[i+1]
            frame_pair = f't{i+1}_t{i+2}'
            output_path = horn_schunk(img1_path, img2_path, frame_pair)
            result_frames.append(output_path)

        result_gif_path = os.path.join(app.root_path, 'static', 'output', 'result_video.gif')
        create_gif_from_frames(result_frames, result_gif_path, fps)
        return render_template('result.html', video_path=result_gif_path)

if __name__ == '__main__':
    app.run(debug=True)

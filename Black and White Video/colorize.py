"""Credits: 
	1. https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py
	2. http://richzhang.github.io/colorization/
	3. https://github.com/richzhang/colorization/
"""

# Import statements

"""
Download the model files: 
	1. colorization_deploy_v2.prototxt:    https://github.com/richzhang/colorization/tree/caffe/colorization/models
	2. pts_in_hull.npy:					   https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
	3. colorization_release_v2.caffemodel: https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

"""
import numpy as np
import argparse
import cv2
import os
# Paths to load the model
DIR = r"C:/Users/Hariharan K/OneDrive/Desktop/codeclasue/Black and White Video"

PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

print(os.path.exists(PROTOTXT))
print(os.path.exists(POINTS))
print(os.path.exists(MODEL))
# ... (previous code remains unchanged)

# Argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
    help="path to input black and white video")
args = vars(ap.parse_args())

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Open the video file
cap = cv2.VideoCapture(args["input"])

# Get the video properties
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(5)
total_frames = int(cap.get(7))  # Total number of frames in the video

# Create VideoWriter object to save the colorized video as .mp4
out = cv2.VideoWriter('colorized_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    # Preprocess the frame for colorization
    scaled = frame.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Colorize the frame
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Write the colorized frame to the output video
    out.write(colorized)

# Release the video capture and writer objects
cap.release()
out.release()

# Close all windows
cv2.destroyAllWindows()

# # Argparser
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True,
# 	help="path to input black and white image")
# args = vars(ap.parse_args())

# # Load the Model
# print("Load model")
# net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
# pts = np.load(POINTS)

# # Load centers for ab channel quantization used for rebalancing.
# class8 = net.getLayerId("class8_ab")
# conv8 = net.getLayerId("conv8_313_rh")
# pts = pts.transpose().reshape(2, 313, 1, 1)
# net.getLayer(class8).blobs = [pts.astype("float32")]
# net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# # Load the input image
# image = cv2.imread(args["image"])
# scaled = image.astype("float32") / 255.0
# lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# resized = cv2.resize(lab, (224, 224))
# L = cv2.split(resized)[0]
# L -= 50

# print("Colorizing the image")
# net.setInput(cv2.dnn.blobFromImage(L))
# ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

# ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

# L = cv2.split(lab)[0]
# colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

# colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
# colorized = np.clip(colorized, 0, 1)

# colorized = (255 * colorized).astype("uint8")

# cv2.imshow("Original", image)
# cv2.imshow("Colorized", colorized)
# cv2.waitKey(0) voice should be retained
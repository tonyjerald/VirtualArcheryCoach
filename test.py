import cv2
import PIL
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from glob import glob

#Converting  MP4 to Frames:
# vidcap = cv2.VideoCapture('original.mp4')
# success, image = vidcap.read()
# count = 0
# while success:
#     cv2.imwrite(f"frames/frame{count}.jpg", image)     # Save frame as JPEG file
#     success, image = vidcap.read()
#     count += 1

count = 3
# Preprocess the Images:
def preprocess_image(file_path, target_size):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    return img_array

# for i in range(count):
#     img_array = preprocess_image(f'frames/frame{i}.jpg', target_size=(224, 224))
#     # print(f"test{i}",img_array)
#     cv2.imwrite(f"output/frame{count}.jpg", img_array)



image_files = glob(f'frames/frame*.jpg')
dataset = np.array([preprocess_image(file, (224, 224)) for file in image_files])
print(dataset)


import subprocess

CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)



asdf


import torch
import urllib.request
import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import clip
from PIL import Image
import os.path
from os import path
import sys

bboxdir = '/content/bboxs'
images = []


#YOLO inference

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# INPUT
imgs = ['https://ultralytics.com/images/zidane.jpg']  # batch of images
inputtext = "elephant"

if path.exists(bboxdir) == False:
    os.mkdir(bboxdir)

# Inference
results = model(imgs)

# Results
# results.print()
# print("-----------------")
# print()
# results.save() # or save()
coordinates = results.pandas().xyxy[0]

# Take image
req = urllib.request.urlopen(imgs[0])
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1)
# cv2.imwrite("base.jpg", img)


#CLIP inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

for index, row in coordinates.iterrows():
    # print()
    # print(row)
    xmin = int(row['xmin'])
    xmax = int(row['xmax'])
    ymin = int(row['ymin'])
    ymax = int(row['ymax'])

    print("-----------------")

    cropped_image = img[ymin:ymax, xmin:xmax]
    cv2.imwrite(bboxdir+"/"+str(index) + ".jpg", cropped_image)
    
for filename in [filename for filename in os.listdir(bboxdir) if filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    image = Image.open(os.path.join(bboxdir, filename)).convert("RGB")
    images.append(preprocess(image))

image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize([inputtext]).cuda()

with torch.no_grad():
    # image_features = model.encode_image(image_input).float()
    # text_features = model.encode_text(text_tokens).float()

    logits_per_image, logits_per_text = model(image_input, text_tokens)
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

print(probs[0])
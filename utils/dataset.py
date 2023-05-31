from PIL import Image
from pathlib import Path

import torch
import time

class Dataset(torch.utils.data.Dataset):

	def __init__(self, data, preprocess, mode):
		self.preprocess = preprocess
		self.img_paths = []
		self.captions = []
		self.bboxes = []
		self.sizes = []

		initial = time.time()

		for index, item in data.iterrows():
			self.img_paths.append(item["image_path"])
			self.captions.append(item["caption"])
			self.bboxes.append(item["bbox"])
			self.sizes.append((item["width"], item["height"]))

	def __len__(self):
		return len(self.captions)

	def __getitem__(self, ind):
		image_path = self.img_paths[ind]
		image = self.preprocess(Image.open(image_path))
		caption = self.captions[ind]
		bbox = self.bboxes[ind]
		size = self.sizes[ind]
		return image, caption, bbox, image_path, size

class BboxDataset(torch.utils.data.Dataset):

	def __init__(self, data, preprocess, mode):
		self.preprocess = preprocess
		self.img_paths = []
		self.captions = []
		self.bboxes = []
		self.ids = []
		self.sizes = []

		initial = time.time()

		for index, item in data.iterrows():
			self.img_paths.append(item["image_path"])
			self.captions.append(item["caption"])
			self.bboxes.append(item["bbox"])
			self.ids.append(item["image_path"])
			self.sizes.append((item["width"], item["height"]))

	def __len__(self):
		return len(self.captions)

	def __getitem__(self, ind):
		image_path = self.img_paths[ind]
		image = self.preprocess(Image.open(image_path))
		caption = self.captions[ind]
		bbox = self.bboxes[ind]
		imgid = self.ids[ind]
		size = self.sizes[ind]
		return image, caption, bbox, size
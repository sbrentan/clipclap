import pickle
from PIL import Image

import pandas as pd
import json
import os
import glob

from utils.config import CFG 

class DataManager():

	def parse_instances():
		instances = json.load(open(CFG.annotation_path + "/instances.json", "rb"))

		images_ist = {}

		for img in instances["images"]:
			images_ist[img["id"]] = {}
			images_ist[img["id"]]["file_name"] = img["file_name"]
			images_ist[img["id"]]["width"] = img["width"]
			images_ist[img["id"]]["height"] = img["height"]

		categories_ist = {}

		for cat in instances["categories"]:
			categories_ist[cat["id"]] = cat["name"]

		annotations_ist = {}

		for ann in instances["annotations"]:
			annotations_ist[ann["id"]] = {}
			annotations_ist[ann["id"]]["image_id"] = ann["image_id"]
			annotations_ist[ann["id"]]["image_name"] = images_ist[ann["image_id"]]["file_name"]
			annotations_ist[ann["id"]]["image_width"] = images_ist[ann["image_id"]]["width"]
			annotations_ist[ann["id"]]["image_height"] = images_ist[ann["image_id"]]["height"]
			annotations_ist[ann["id"]]["bbox"] = ann["bbox"]
			annotations_ist[ann["id"]]["category"] = categories_ist[ann["category_id"]]

		return annotations_ist

	def crop_image_and_save(bbox, image_path, output_path):
		xmin = bbox[0]
		xmax = xmin + bbox[2] # width
		ymin = bbox[1]
		ymax = ymin + bbox[3] # height
	   	
		img = Image.open(image_path)
		cropped_image = img.crop((xmin,ymin,xmax,ymax)) 

		cropped_image.save(output_path)

	def parse_data(force_crop=False):

		if not os.path.exists(CFG.cropped_dataset_path):
			os.makedirs(CFG.cropped_dataset_path)
		
		if os.listdir(CFG.cropped_dataset_path) == []:
			force_crop = True

		if force_crop:
			files = glob.glob(CFG.cropped_dataset_path + "/*")
			for f in files:
				os.remove(f)
			print("[CROP] Cropping images")

		annotations_ist = DataManager.parse_instances()

		refs = pickle.load(open(CFG.annotation_path + "/refs(umd).p", "rb"))
		refs = pd.DataFrame.from_dict(refs)

		data = {"image_name": [], "image_path": [], "caption_number": [], "caption": [], "split": []}

		for index, annotation in refs.iterrows():

			annId = annotation.ann_id

			cropped_path = (CFG.cropped_dataset_path + "/" + annotations_ist[annId]["image_name"]).replace(".jpg", "_" + str(annId) + ".jpg")

			if force_crop:
				image_path = CFG.image_path + "/" + annotations_ist[annId]["image_name"]
				DataManager.crop_image_and_save(annotations_ist[annId]["bbox"], image_path, cropped_path)

			for i in range(len(annotation.sentences)):

				data["image_name"].append(annotations_ist[annId]["image_name"])
				data["image_path"].append(cropped_path)
				data["caption_number"].append(i)
				data["caption"].append(annotation.sentences[i]["sent"])
				data["split"].append(annotation.split)

		data = pd.DataFrame.from_dict(data)
	 
		if force_crop:
			print("[CROP] Cropped images saved in {}".format(CFG.cropped_dataset_path))

		train_data = data.loc[data["split"] == "train"]
		test_data = data.loc[data["split"] == "test"]
		val_data = data.loc[data["split"] == "val"]

		return train_data, test_data, val_data

	def parse_data_bbox():

		annotations_ist = DataManager.parse_instances()

		refs = pickle.load(open(CFG.annotation_path + "/refs(umd).p", "rb"))
		refs = pd.DataFrame.from_dict(refs)

		data = {"image_name": [], "image_path": [], "caption_number": [], "caption": [],
				"split": [], "bbox": [], "width": [], "height": []}

		for index, annotation in refs.iterrows():

			annId = annotation.ann_id

			image_path = CFG.image_path + "/" + annotations_ist[annId]["image_name"]

			for i in range(len(annotation.sentences)):

				data["image_name"].append(annotations_ist[annId]["image_name"])
				data["image_path"].append(image_path)
				data["caption_number"].append(i)
				data["caption"].append(annotation.sentences[i]["sent"])
				data["split"].append(annotation.split)
				data["bbox"].append(annotations_ist[annId]["bbox"])
				data["width"].append(annotations_ist[annId]["image_width"])
				data["height"].append(annotations_ist[annId]["image_height"])

		data = pd.DataFrame.from_dict(data)

		train_data = data.loc[data["split"] == "train"]
		test_data = data.loc[data["split"] == "test"]
		val_data = data.loc[data["split"] == "val"]

		return train_data, test_data, val_data
import torch
import os

class CFG:
	#debug = False
	image_path = "./data/refcocog/images"
	annotation_path = "./data/refcocog/annotations"
	cropped_dataset_path = "./data/cropped_dataset"

	# Adam & Scheduler
	lr = 0.01
	weight_decay = 1e-3
	patience = 1
	factor = 0.8

	# model training
	batch_size = 16
	epochs = 100
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	train_samples = 8000
	test_samples = 2400
	
	model_name = "RN50" # resnet50
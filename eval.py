import clip
import torch
from PIL import Image, ImageDraw

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from utils.data_manager import DataManager
from models.clipclap import ClipClap
from utils.dataset import BboxDataset, Dataset
from utils.config import CFG
from utils.loss import Loss

# asdf

class Evaluation():

	def load_clip(modelPath):
		model, preprocess = clip.load(CFG.model_name, device=CFG.device, jit=False)
		checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))

		model.load_state_dict(checkpoint['model_state_dict'])
	 
		print("[CLIP] Model [{}] <-> Batch_Size: {}, Train Samples: {}, Test Samples: {}, Loss: {:.5f}, Accuracy: {:.2f}".format(
			modelPath, 
			checkpoint['batch_size'], 
			checkpoint['train_samples'], 
			checkpoint['test_samples'], 
			checkpoint['loss'], 
			checkpoint['accuracy']
		))
	 
		return model, preprocess

	def load_clipclap(clip_model, modelPath):
		checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
		model = ClipClap(clip_model)
		model.load_state_dict(checkpoint['model_state_dict'])
	 
		# print("[CLIP] Model [{}] <-> Batch_Size: {}, Train Samples: {}, Test Samples: {}, Loss: {:.5f}, Accuracy: {:.2f}".format(
		# 	modelPath, 
		# 	checkpoint['batch_size'], 
		# 	checkpoint['train_samples'], 
		# 	checkpoint['test_samples'], 
		# 	checkpoint['loss'], 
		# 	checkpoint['accuracy']
		# ))
	 
		return model

	def save_model(epoch, model, loss, accuracy, name):
		
		if name == "best.pt":
			path = "./models/" + name
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'loss': loss,
				'accuracy': accuracy,
			}, path)
			print(name + " saved in " + path)
		
		path = "/content/drive/MyDrive/Unitn/LM/Assignments/Deep Learning/models"

		if not os.path.exists(path):
			path = "/content/drive/MyDrive"

		path += "/" + name

		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'batch_size': CFG.batch_size,
			'train_samples': CFG.train_samples,
			'test_samples': CFG.test_samples,
			'loss': loss,
			'accuracy': accuracy,
		}, path)
		print(name + " saved in " + path)

	def start():
		model_path = "data\\checkpoint.pt"
		clip_model, preprocess = Evaluation.load_clip(model_path)
		clipclap_model = Evaluation.load_clipclap(clip_model, "data\\clipclap_13_11870487056245336.pt")

		bs = 1
		
		train_data, test_data, eval_data = DataManager.parse_data_bbox()
		train_dataset = Dataset(train_data, preprocess, mode="train")
		# test_dataset = BboxDataset(test_data, preprocess, mode="test")

		train_samples = 1
		# test_samples = 100
		train_sampler = RandomSampler(train_dataset, num_samples=train_samples, replacement=True)
		# test_sampler = RandomSampler(test_dataset, num_samples=CFG.test_samples)

		train_dataloader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler, drop_last=True)
		# test_dataloader = DataLoader(test_dataset, batch_size=bs, sampler=test_sampler, drop_last=True)

		clip_model.eval()
		clipclap_model.eval()

		with torch.no_grad():

			for ind, batch in enumerate(train_dataloader):
				images, captions, bbox, path, size = batch
				print(images, captions, size)
				bbox = torch.stack(bbox).transpose(0, 1)
				size = torch.stack(size).repeat(2, 1).transpose(0, 1)

				pred_bbox = clipclap_model(images, captions)
				# temp_loss = nn.functional.l1_loss(pred_bbox, torch.cat(bboxes).reshape(bs, 4)).item()
				# loss2 += temp_loss
				# size = torch.cat([torch.cat(size), torch.cat(size)]).cpu().detach().numpy()
				print(pred_bbox)
				print(bbox)
				print(size)
				print(bbox/size)
				loss_dict = Loss.clipclap_loss(pred_bbox, bbox/size)
				print(loss_dict)
				loss = sum(loss_dict[k] for k in loss_dict.keys())
				print("[ LOSS ]: ", loss)

				pred_bbox = pred_bbox.cpu().detach().numpy()[0]
				size = size.cpu().detach().numpy()[0]
				bbox = bbox.cpu().detach().numpy()[0]

				print(pred_bbox)
				print(size)
				pred_bbox *= size
				print(pred_bbox)
				img_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[2]+pred_bbox[0], pred_bbox[3]+pred_bbox[1]]
				gt_bbox = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
				print(img_bbox)
				img = Image.open(path[0])
				draw = ImageDraw.Draw(img)
				draw.rectangle(img_bbox, outline="red", width=2)
				draw.rectangle(gt_bbox, outline="green", width=2)
				img.show()
				print(captions)

			



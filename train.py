import clip
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from utils.data_manager import DataManager
from models.clipclap import ClipClap
from utils.dataset import BboxDataset
from utils.config import CFG
from utils.loss import Loss

# asdf

class Training():

	def load_model(modelPath):
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
		
		path = "data"

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
		clip_model, preprocess = Training.load_model(model_path)

		bs = CFG.batch_size
		
		train_data, test_data, _ = DataManager.parse_data_bbox()
		# print(train_data.iloc[0])
		# print()
		# print(train_data.iloc[1])
		# sys.exit(0)
		train_dataset = BboxDataset(train_data, preprocess, mode="train")
		test_dataset = BboxDataset(test_data, preprocess, mode="test")

		# train_samples = 32
		# test_samples = 100
		train_sampler = RandomSampler(train_dataset, num_samples=CFG.train_samples, replacement=True)
		# test_sampler = RandomSampler(test_dataset, num_samples=CFG.test_samples)

		train_dataloader = DataLoader(train_dataset, batch_size=bs, sampler=train_sampler, drop_last=True)
		# test_dataloader = DataLoader(test_dataset, batch_size=bs, sampler=test_sampler, drop_last=True)

		clipclap_model = ClipClap(clip_model).to(CFG.device)
		# print(clip_model)
		# exit()

		clip_model.eval()

		optimizer = optim.SGD(clipclap_model.parameters(), lr=CFG.lr)

		eval_loss = float("inf")
		for e in range(CFG.epochs):
		
			clipclap_model.train()
			for ind, batch in enumerate(train_dataloader):
				break
				optimizer.zero_grad()
				images, captions, bboxes, size = batch
				images = images.to(CFG.device).float()
				bboxes = torch.stack(bboxes).transpose(0, 1).to(CFG.device)
				size = torch.stack(size).repeat(2, 1).transpose(0, 1).to(CFG.device)
				# print(images)
				# print(captions)
				# print(bboxes)
				# print(size)
				# print(bboxes/size)
				# print(imgids)
				# print(torch.cat(bboxes).reshape(bs, 4))
				# print()
				# exit()

				# pred_bbox = clipclap_model(image, caption)
				pred_bbox = clipclap_model(images, captions)
				# loss = nn.functional.l1_loss(pred_bbox, bboxes)
				temp_bbox = pred_bbox[:, :2]
				pred_bbox[:, 2:] += temp_bbox
				temp_bbox = bboxes[:, :2]
				bboxes[:, 2:] += temp_bbox
				loss_dict = Loss.clipclap_loss(pred_bbox, bboxes/size)
				# print(pred_bbox)
				# print(bboxes)
				# print(size)
				# print(bboxes/size)
				# exit()
				loss = sum(loss_dict[k] for k in loss_dict.keys())

				# print("================================================")
				# print(f"Step {ind}:", loss.item())


				loss.backward()



				# print(pred_bbox)
				# print([max(i) - min(i) for i in pred_bbox.transpose(0, 1).cpu().detach().numpy()])
				train_bar.set_postfix(batch_train_loss=loss.item(), 
					max_diff=[max(i) - min(i) for i in pred_bbox.transpose(0, 1).cpu().detach().numpy()])

				# clipclap_model.float()
				optimizer.step()

				# exit()
				# print(pred_bbox, bboxes)

			print(f'epoch {e} finished')
			clipclap_model.eval()

			with torch.no_grad():
				loss2 = 0
				for ind2, batch2 in enumerate(test_dataloader):
					images, captions, bboxes, size = batch2
					images = images.to(CFG.device)
					bboxes = torch.stack(bboxes).transpose(0, 1).to(CFG.device)
					size = torch.stack(size).repeat(2, 1).transpose(0, 1).to(CFG.device)

					pred_bbox = clipclap_model(images, captions)
					temp_bbox = pred_bbox[:, :2]
					pred_bbox[:, 2:] += temp_bbox
					temp_bbox = bboxes[:, :2]
					bboxes[:, 2:] += temp_bbox



					# temp_loss = nn.functional.l1_loss(pred_bbox, gt_bbox).item()
					loss_dict = Loss.clipclap_loss(pred_bbox, bboxes/size)
					temp_loss = sum(loss_dict[k] for k in loss_dict.keys())
					# print("temp loss:", temp_loss)
					loss2 += temp_loss

				loss2 /= (test_samples // bs)
				print('TEST LOSS:', loss2, eval_loss)
				# if(loss2 < eval_loss):
				eval_loss = loss2
				Training.save_model(e+1, clipclap_model, loss2, -1, "clipclap_"+str(e+1)+"_"+str(round(loss2, 3))+".pt")



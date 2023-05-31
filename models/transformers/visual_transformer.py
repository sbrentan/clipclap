import torch
import torch.nn as nn
from torchvision.models import resnet50

from models.embeddings.position_encoding import build_position_encoding
from models.visual_clip.transformer.custom_transformer import TransformerEncoder, TransformerEncoderLayer

from utils.config import CFG

# import sys

class VisualTransformer(nn.Module):

	def __init__(self, clip_model, hidden_dim=256, nheads=8,
				 num_encoder_layers=6, num_decoder_layers=6):
		super().__init__()

		# create ResNet-50 backbone
		# self.backbone = resnet50()
		# del self.backbone.fc
		# print(self.backbone)
		# print('------------------------------------------------------')
		# print(clip_model)
		# sys.exit(0)
		self.clip_model = clip_model
		self.hidden_dim = hidden_dim

		self.pos_embedding = build_position_encoding() #add to images

		# create conversion layer
		self.conv = nn.Conv2d(2048, hidden_dim, 1)

		# create a default PyTorch transformer
		# self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
		# self.transformer = CTransformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
		encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, 2048,
                                                0.1, "relu", False)
		encoder_norm = None
		self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

		# prediction heads, one extra class for predicting non-empty slots
		# note that in baseline DETR linear_bbox layer is 3-layer MLP
		# self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
		# self.linear_bbox = nn.Linear(hidden_dim, 4)

		# output positional encodings (object queries)
		self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

		# num_queries = 100
		# self.query_embed = nn.Embedding(num_queries, hidden_dim)

		# spatial positional encodings
		# note that in baseline DETR we use sine positional encodings
		self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
		self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))



	def forward(self, img):
		# propagate inputs through ResNet-50 up to avg-pool layer

		bs = img.shape[0]
		# print('bs:', bs)
		

		# print('visual shape 0:', img.shape)

		x = self.clip_model.visual.conv1(img)
		x = self.clip_model.visual.bn1(x)
		x = self.clip_model.visual.relu1(x)
		x = self.clip_model.visual.conv2(x)
		x = self.clip_model.visual.bn2(x)
		x = self.clip_model.visual.relu2(x)
		x = self.clip_model.visual.conv3(x)
		x = self.clip_model.visual.bn3(x)
		x = self.clip_model.visual.relu3(x)

		x = self.clip_model.visual.avgpool(x)

		x = self.clip_model.visual.layer1(x)
		x = self.clip_model.visual.layer2(x)
		x = self.clip_model.visual.layer3(x)
		x = self.clip_model.visual.layer4(x) #([16, 2048, 7, 7])


		# x = self.clip_model.visual.attnpool(x) #([16, 2048])
		# print("x.shape:", x.shape)
		# # x_pos = self.pos_embedding(h.permute(0, 2, 3, 1))
		# x_pos = self.pos_embedding(x)
		# x_pos = x_pos.flatten(2).permute(2, 0, 1)
		src = self.conv(x)
		# print('visual shape 1:', x.shape)
		# convert from 2048 to 256 feature planes for the transformer
		# print('visual shape 2:', h.shape)
		# # construct positional encodings
		# H, W = h.shape[-2:]
		# pos = torch.cat([
		# 	self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
		# 	self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
		# ], dim=-1).flatten(0, 1).unsqueeze(1).repeat(1, bs, 1)
		# # print('visual shape 3:', pos.shape) # (49, 1, 256)
		# # propagate through the transformer
		# h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1), # (49, 1, 256)
		# 					 x_pos)#self.query_pos.unsqueeze(1).repeat(1, bs, 1)) #(100, 256)


		# original detr transformer
		mask_shape = (src.shape[0], src.shape[2], src.shape[3])
		mask = torch.zeros(mask_shape, dtype=torch.bool).to(CFG.device)
		pos = self.pos_embedding(x)
		# print("src.shape:", src.shape)
		# print("mask.shape:", mask.shape)
		# print("pos.shape:", pos.shape)
		# print("self.query_embed.weight.shape:", self.query_embed.weight.shape)

		bs, c, h, w = src.shape
		src = src.flatten(2).permute(2, 0, 1)
		pos = pos.flatten(2).permute(2, 0, 1)
		mask = mask.flatten(1)
		# mask = None

		# print("src2:", src.shape)
		# print("pos2:", pos.shape)
		# print("mask2:", mask.shape)

		h = self.transformer(src, src_key_padding_mask=mask, pos=pos)
		#pos added to src in encoder, query_embed added to memory(encoder result) in decoder

		# print(hs.shape)

		# exit()

		# memory.permute(1, 2, 0).view(bs, c, h, w)

		# print('transpose:',h.shape) #(100, 1, 256)
		# h = h.transpose(0, 1)
		# print('transpose:',h.shape) #(1, 100, 256)

		# print('transformer d_model:', self.transformer.d_model)

		# print('visual shape 4:', h.shape)

		return h, mask, pos
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, List

import clip
import copy
import math

from models.transformers.visual_transformer import VisualTransformer
from models.embeddings.bbox_mlp import BboxMLP, MLP
from utils.config import CFG


from transformers import DistilBertTokenizerFast, DistilBertModel

DEBUG = False


class ClipClap(nn.Module):
	def __init__(self, clip_model, hidden_dim=256, nheads=8,
				num_encoder_layers=6, num_decoder_layers=6):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.clip_model = clip_model
		self.vtrans = VisualTransformer(clip_model)

		# self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
		# self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		# self.txt_transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
		# self.img_transformer = nn.Transformer()

		#print(self.clip_model.visual.conv1)
		# exit()

		self.visu_proj = nn.Linear(256, hidden_dim)
		self.text_proj = nn.Linear(self.clip_model.text_projection.shape[1] // 16, hidden_dim)
		# self.text_proj = nn.Linear(768, hidden_dim)

		# enc_layer = nn.TransformerEncoderLayer(hidden_dim, 8)
		# self.tvtrans = nn.TransformerEncoder(enc_layer, 6)
		self.tvtrans = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

		divisor = 32 #if args.dilation else 32
		self.num_visu_token = int((224 / divisor) ** 2) #args.imsize
		#print(self.num_visu_token)

		self.num_text_token = 16 #args.max_query_len
		num_total = self.num_visu_token + self.num_text_token + 1
		self.vl_pos_embedding = nn.Embedding(num_total, hidden_dim)
		self.reg_token = nn.Embedding(1, hidden_dim)

		self.bbox_mlp = BboxMLP(hidden_dim, hidden_dim, 4, 3)



		self.img_mlp = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
		self.txt_mlp = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
		self.attn1 = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
		self.attn2 = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
		# self.attn3 = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
		self.attn3 = MHAttentionRPE(d_model=256, h=8, dropout=0.1,
				pos_x_range=[-20, 20],
				pos_y_range=[-20, 20],
				pos_index_offset=20)
		self.tf_pow = 2.0
		self.tf_scale = nn.Parameter(torch.Tensor([1.0]))
		self.tf_sigma = nn.Parameter(torch.Tensor([0.5]))
		self.norm_text_cond_img = nn.LayerNorm(hidden_dim)
		self.norm_img = nn.LayerNorm(hidden_dim)
		num_queries = 1
		query_dim = 256
		self.vis_query_embed = nn.Embedding(num_queries, query_dim)
		self.text_query_embed = nn.Embedding(num_queries, query_dim)
		decoder_layer = MultiStageDecoderLayer(256)
		self.decoder_layers = _get_clones(decoder_layer, 6)
		self.norm = nn.LayerNorm(256)


	def with_pos_embed(self, tensor, pos):
		return tensor if pos is None else tensor + pos

	def forward(self, img, text):
		bs = img.shape[0]

		#where to put cuda() or cpu()?

		#print(img.shape)
		# #print(text.shape)
		
		#pos enc
		# #print('input_img shape 0:', img.shape)
		# input_img = img.unsqueeze(0)#.cuda()
		# #print('input_img shape:', input_img.shape)
		enc_img, img_mask, pos_embed = self.vtrans(img)
		# self.img_transformer(enc_img)    # already in vtrans

		if DEBUG: print('enc_img shape:', enc_img.shape)

		
		if False:
			train_encodings = self.tokenizer(text, truncation=True, padding=True)
			maxlen = max([len(train_encodings['input_ids'][i]) for i in range(len(train_encodings['input_ids']))])
			# print(maxlen)
			for i in range(len(train_encodings['input_ids'])):
				for j in range(50 - maxlen):
					train_encodings['input_ids'][i].append(0)
					train_encodings['attention_mask'][i].append(0)
			outputs = self.bert(torch.tensor(train_encodings["input_ids"]), attention_mask=torch.tensor(train_encodings["attention_mask"]))
			enc_txt = outputs.last_hidden_state
			# enc_txt = all_encoder_layers[11]
			# print(outputs.last_hidden_state.shape)
			# exit()
		else:
			#pos enc
			input_txt = clip.tokenize(text).to(CFG.device)
			#print('tokenized:',input_txt.shape)
			enc_txt = self.clip_model.encode_text(input_txt) # remove last layer ?
			enc_txt = enc_txt.reshape(bs, 16, self.clip_model.text_projection.shape[1] // 16)
			# self.txt_transformer(enc_txt)	


		#	COMPUTED enc_img and enc_txt

		if False:
			# x = self.clip_model.token_embedding(input_txt).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
			# x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
			# x = x.permute(1, 0, 2)  # NLD -> LND
			# x = self.clip_model.transformer(x)
			# x = x.permute(1, 0, 2)  # LND -> NLD
			# x = self.clip_model.ln_final(x).type(self.clip_model.dtype)


			# #print(self.clip_model.token_embedding.weight.shape)
			# #print(input_txt)

			# #print(input_txt.argmax(dim=-1))


			# r = x[torch.arange(x.shape[0]), input_txt.argmax(dim=-1)]
			# #print(r.shape)
			# #print(self.clip_model.text_projection.shape)
			# r = r @ self.clip_model.text_projection
			# #print(r.shape)

			# #print(x)

			

			# #print("xshape:",x.shape)



			#print('enc_txt shape:', enc_txt.shape)

			#print('enc_txt reshaped:', enc_txt.reshape(bs, 16, self.clip_model.text_projection.shape[1] // 16).shape)

			proj_img = self.visu_proj(enc_img)
			proj_txt = self.text_proj(enc_txt)

			
			if DEBUG: print('proj_img shape:', proj_img.shape)
			if DEBUG: print('proj_txt shape:', proj_txt.shape)

			# #print(self.reg_token)
			# s = self.reg_token.weight.unsqueeze(1)
			# #print(s.shape)
			# s = s.repeat(1, bs, 1)
			tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
			if DEBUG: print('reg_token shape:', tgt_src.shape)

			#print()
			#print('reg_token shape:', tgt_src.shape)
			#print('proj_txt shape:', proj_txt.permute(1,0,2).shape)
			#print('proj_img shape:', proj_img.permute(1,0,2).shape)

			vl_src = torch.cat([tgt_src, proj_txt.permute(1,0,2), proj_img.permute(1,0,2)], dim=0)

			#print('vl_src shape:', vl_src.shape)

			#print("vl_pos_embedding shape:", self.vl_pos_embedding.weight.shape)
			vl_pos = self.vl_pos_embedding.weight.unsqueeze(1).repeat(1, bs, 1)

			#print('vl_pos shape:', vl_pos.shape)

			#print('unsqueezed proj_txt:', proj_txt.unsqueeze(1).shape)


			# create tensor length (1+img_len+txt_len)
			#print('sum:', (vl_pos + vl_src).shape)
			enc_fusion = self.tvtrans(vl_pos + vl_src)

			if DEBUG: print('enc_fusion:', enc_fusion.shape)

			if DEBUG: print('enc_fusion[0] shape:', enc_fusion[0])

			hs = enc_fusion[0]

			#print('bbox shape:', bbox.shape)

		else:
			proj_txt = self.text_proj(enc_txt)
			proj_txt = proj_txt.permute(1, 0, 2)

			img_feat = enc_img
			img_key_padding_mask = img_mask
			word_feat = proj_txt
			word_key_padding_mask = torch.zeros((word_feat.shape[1], word_feat.shape[0]), dtype=torch.bool).to(CFG.device)
			word_pos = None

			orig_img_feat = img_feat# not using + img_pos

			# visual-linguistic verification
			img_query = enc_img #False
			if DEBUG: print("img_query.shape", img_query.shape)
			if DEBUG: print("img_key_padding_mask.shape", img_key_padding_mask.shape)
			if DEBUG: print("word_feat.shape", word_feat.shape)
			if DEBUG: print("word_key_padding_mask.shape", word_key_padding_mask.shape)
			text_info = self.attn1(#MultiheadAttention
				query=img_query, key=word_feat,
				value=word_feat, key_padding_mask=word_key_padding_mask)[0]

			text_embed = self.txt_mlp(text_info)
			img_embed = self.img_mlp(img_feat)
			verify_score = (F.normalize(img_embed, p=2, dim=-1) *
							F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
			verify_score = self.tf_scale * \
						   torch.exp( - (1 - verify_score).pow(self.tf_pow) \
							/ (2 * self.tf_sigma**2))

			# language-guided context encoder
			text_cond_info = self.attn2(#MultiheadAttention
				query=img_feat, key=word_feat,
				value=word_feat, key_padding_mask=word_key_padding_mask)[0]

			q = k = img_feat + text_cond_info
			text_cond_img_ctx = self.attn3(#MHAttentionRPE
				query=q, key=k, value=img_feat, key_padding_mask=img_key_padding_mask)[0]

			# discriminative feature
			fuse_img_feat = (self.norm_img(img_feat) +
							 self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

			img_feat = torch.cat([orig_img_feat, fuse_img_feat], dim=-1)

			if DEBUG: print(img_feat.shape)

			vis_query_embed = self.vis_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
			text_query_embed = self.text_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

			# Initial target query
			vis_query = torch.zeros_like(vis_query_embed)

			for idx, layer in enumerate(self.decoder_layers):
				vis_query = layer(vis_query, vis_query_embed, text_query_embed,
                              img_feat, img_key_padding_mask, pos_embed,
                              word_feat, word_key_padding_mask, None, idx)

			output = self.norm(vis_query)
			hs = output[0]#output.unsqueeze(0)

		if DEBUG: print(hs.shape) #[1, 1, 4, 256] invece di [6, 4, 1, 256]
		bbox = self.bbox_mlp(hs)

		if DEBUG: print(bbox.shape)

		# exit()


		return bbox
		




class MHAttentionRPE(nn.Module):
	''' With relative position embedding '''
	def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
				 pos_x_range=[-20, 20], pos_y_range=[-20, 20], pos_index_offset=20,
				 learnable_pos_embed=False):
		super().__init__()
		self.d_k = d_model // h
		self.h = h
		self.scaling = float(self.d_k) ** -0.5
		self.return_raw_attention = return_raw_attention

		self.in_proj_weight = nn.Parameter(torch.Tensor(3 * d_model, d_model))
		self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
		self.out_proj = nn.Linear(d_model, d_model, bias=True)

		self.attn = None
		# self.dropout = nn.Dropout(p=dropout)
		self.dropout_p = dropout
		self._reset_parameters()

		self.learnable_pos_embed = learnable_pos_embed
		if learnable_pos_embed:
			self.pos_x = nn.Embedding(pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
			self.pos_y = nn.Embedding(pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
		else:
			pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
												   x_range=pos_x_range, y_range=pos_y_range)
			self.register_buffer('pos_x', pos_x) # [x_range, C]
			self.register_buffer('pos_y', pos_y) # [y_range, C]

		self.pos_index_offset = pos_index_offset

	def _reset_parameters(self):
		nn.init.xavier_uniform_(self.in_proj_weight)
		nn.init.constant_(self.in_proj_bias, 0.)
		nn.init.constant_(self.out_proj.bias, 0.)


	def forward(self, query, key, value, key_padding_mask=None):
		tgt_len, bs, dim = query.size()
		src_len, _, dim = key.size()

		weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
		weight_k, bias_k = self.in_proj_weight[dim:dim*2], self.in_proj_bias[dim:dim*2]
		weight_v, bias_v = self.in_proj_weight[dim*2:], self.in_proj_bias[dim*2:]

		q = query.matmul(weight_q.t()) + bias_q
		k = key.matmul(weight_k.t()) + bias_k
		v = value.matmul(weight_v.t()) + bias_v

		q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)  # [bs*h, tgt_len, dim//h]
		k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)  # [bs*h, dim//h, src_len], To calculate qTk (bmm)
		v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

		q = q * self.scaling
		attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

		### compute the relative positions
		bs, HW = key_padding_mask.size()
		# assert (HW == 400) and (HW == tgt_len)
		# img_mask = ~key_padding_mask.view(bs, 20, 20)
		img_mask = ~key_padding_mask.view(bs, 7, 7)
		yy = img_mask.cumsum(1, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
		xx = img_mask.cumsum(2, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
		diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
		diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]
		if self.learnable_pos_embed:
			k_posy = self.pos_y.weight.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
			k_posx = self.pos_x.weight.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
		else:
			k_posy = self.pos_y.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
			k_posx = self.pos_x.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
		k_posy = k_posy.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
						reshape(-1, bs * self.h, dim//self.h).permute(1, 2, 0)  # [bs*h, dim//h, y_range]
		k_posx = k_posx.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
						reshape(-1, bs * self.h, dim//self.h).permute(1, 2, 0)  # [bs*h, dim//h, x_range]
		posy_attn_weights = torch.bmm(q, k_posy).view(bs, self.h, HW, -1)  # [bs, h, HW, y_range]
		posx_attn_weights = torch.bmm(q, k_posx).view(bs, self.h, HW, -1) # [bs, h, HW, x_range]
		diff_yy_idx = diff_yy[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset
		diff_xx_idx = diff_xx[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset

		posy_attn_weights = torch.gather(posy_attn_weights, -1, diff_yy_idx.long()) # [bs, h, HW, HW]
		posx_attn_weights = torch.gather(posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]
		pos_attn_weights = (posy_attn_weights + posx_attn_weights).view(bs*self.h, HW, -1)
		attn_weights = attn_weights + pos_attn_weights


		if key_padding_mask is not None:
			attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
			attn_weights = attn_weights.masked_fill(
				key_padding_mask.unsqueeze(1).unsqueeze(2),  # [bs, 1, 1, src_len]
				float('-inf')
			)
			attn_weights = attn_weights.view(-1, tgt_len, src_len)
		raw_attn_weights = attn_weights
		attn_weights = attn_weights.softmax(dim=-1)
		attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
		attn_output = torch.bmm(attn_weights, v)
		self.attn = attn_weights

		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
		attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
		if self.return_raw_attention:
			return attn_output, raw_attn_weights
		return attn_output, attn_weights



def position_embedding_sine(num_pos_feats=64, temperature=10000, normalize=False, scale=None,
							x_range=[-20, 20], y_range=[-20, 20], device=None):

	if scale is not None and normalize is False:
		raise ValueError("normalize should be True if scale is passed")
	if scale is None:
		scale = 2 * math.pi

	x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device) #
	y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
	if normalize:
		eps = 1e-6
		y_embed = y_embed / (y_embed[-1] + eps) * scale
		x_embed = x_embed / (x_embed[-1] + eps) * scale

	dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
	dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

	pos_x = x_embed[:, None] / dim_t
	pos_y = y_embed[:, None] / dim_t
	pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
	pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
	return pos_x, pos_y


class MultiStageDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, img_feat_chunk_num=2):
        super().__init__()
        # args = word_attn_args.copy()
        self.word_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        # args = img_attn_args.copy()
        self.img_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        # Implementation of Feedforward model
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))

        self.norm = _get_clones(nn.LayerNorm(d_model), 3)
        self.dropout = _get_clones(nn.Dropout(dropout), 3)

        self.img_feat_chunk_num = img_feat_chunk_num

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, vis_query, vis_query_pos, text_query_pos,
                img_feat=None, img_key_padding_mask=None, img_pos=None,
                word_feat=None, word_key_padding_mask=None, word_pos=None, layer_idx=None):

        if self.img_feat_chunk_num > 1:
            img_feat_srcs = img_feat.chunk(self.img_feat_chunk_num, dim=-1)
            img_feat_k = img_feat_srcs[1]#fuse_img_feat
            img_feat_v = img_feat_srcs[0]#orig_img_feat
        else:
            img_feat_k = img_feat_v = img_feat

        # Aggregate linguistic info about the object
        text_info = self.word_attn(query=self.with_pos_embed(vis_query, vis_query_pos),
                                   key=self.with_pos_embed(word_feat, word_pos),
                                   value=word_feat, key_padding_mask=word_key_padding_mask)[0]
        text_query = self.norm[0](self.dropout[0](text_info))

        # Gather visual feats based on the linguistic info
        vis_info = self.img_attn(query=self.with_pos_embed(text_query, text_query_pos),
                                 key=self.with_pos_embed(img_feat_k, img_pos),
                                 value=img_feat_v, key_padding_mask=img_key_padding_mask)[0]

        vis_query = self.norm[1](vis_query + self.dropout[1](vis_info))
        vis_query = self.norm[2](vis_query + self.dropout[2](self.ffn(vis_query)))

        return vis_query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

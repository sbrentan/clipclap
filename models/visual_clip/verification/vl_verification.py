import torch
import torch.nn as nn
import torch.nn.functional as F


class VLVerification(nn.Module):
    def __init__(self, visual_feature_size, text_embedding_size, num_heads, hidden_size):
        super(VLVerification, self).__init__()
        self.visual_projection = nn.Linear(visual_feature_size, hidden_size)
        self.vl_projection = nn.Linear(text_embedding_size, hidden_size)
        self.multihead_attention = MultiHeadAttention(hidden_size, num_heads)
        # self.fc = nn.Linear(hidden_size, hidden_size)

        self.param_a = nn.Parameter(torch.Tensor([0.5]))  # Learnable parameter
        self.param_b = nn.Parameter(torch.Tensor([0.2]))  # Learnable parameter

    def forward(self, visual_features, text_embeddings):
        # visual_query = self.visual_projection(visual_features)
        # text_key = self.text_projection(text_embeddings)
        # text_value = self.text_projection(text_embeddings)

        attended_text = self.multihead_attention(visual_features, text_embeddings)#text_key, text_value)
        vl_projection = self.vl_projection(attended_text)
        vl_norm = F.normalize(vl_projection, p=2, dim=-1)

        visual_projection = self.visual_projection(visual_features)
        visual_norm = F.normalize(visual_projection, p=2, dim=-1)

        # Compute exponential with learnable parameters
        pw_verification = vl_norm * visual_norm.transpose(0, 1)
        pw_verification = self.param_a * torch.exp( (1 - pw_verification)**2 / 2*(self.param_b**2))

        # projected_verification = self.fc(pixel_wise_verification)
        return pw_verification


# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_size = hidden_size // num_heads

#         self.query_projection = nn.Linear(hidden_size, hidden_size)
#         self.key_projection = nn.Linear(hidden_size, hidden_size)
#         self.value_projection = nn.Linear(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, hidden_size)

#     def forward(self, query, key, value):
#         batch_size, seq_len, _ = query.size()

#         query = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         key = self.key_projection(key).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         value = self.value_projection(value).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
#         attention_weights = F.softmax(scores, dim=-1)
#         attended_values = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#         x = self.fc(attended_values)
#         return x
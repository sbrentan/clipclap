import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        # self.positional_encoding = PositionalEncoding(hidden_size)
        # self.encoder_layers = nn.ModuleList(
        #     [EncoderLayer(hidden_size, num_heads) for _ in range(num_layers)]
        # )
        # self.fc = nn.Linear(hidden_size, input_size)
        self.attention1 = MultiHeadAttention(hidden_size, num_heads)
        self.attention2 = CustomMHA(hidden_size, num_heads)

    def forward(self, vis_feat, txt_embed):
        # x = self.embedding(x)
        # x = self.positional_encoding(x)

        # for layer in self.encoder_layers:
        #     x = layer(x)

        # x = F.softmax(self.fc(x), dim=-1)
        feat_map = self.attention1(query=vis_feat, key=txt_embed, value=txt_embed)
        x = feat_map + vis_feat
        x = self.attention2(query=x, key=x, value=vis_feat)
        return x


# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super(EncoderLayer, self).__init__()
#         self.attention = MultiHeadAttention(hidden_size, num_heads)
#         self.feed_forward = FeedForward(hidden_size)

#     def forward(self, x):
#         x = self.attention(x)
#         x = self.feed_forward(x)
#         return x

class CustomMHA(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.r_temp = torch.arange(self.key.weight.shape[1])

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        query = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # mat1 = query
        # mat2 = key.transpose(-2, -1)
        mat1 = query.transpose(1, 2)
        mat2 = key + self.key.weight.transpose(1, 2) * 

        scores = torch.matmul(mat1, mat2) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = self.fc(attended_values)
        return x


class PositionalEncoding(nn.Module): #sinusoidal positional encodings
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.hidden_size = hidden_size

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


# class FeedForward(nn.Module):
#     def __init__(self, hidden_size, dropout=0.1):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
#         self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x


# class PositionalEncoding(nn.Module):
#     def __init__(self, hidden_size, max_seq_len=5000):
#         super(PositionalEncoding, self).__init__()
#         position = torch.arange(0, max_seq_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size

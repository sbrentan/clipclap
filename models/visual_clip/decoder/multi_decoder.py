import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hidden_size=256, num_heads=8, num_layers=6, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        # self.positional_encoding = PositionalEncoding(hidden_size)
        # self.dropout = nn.Dropout(dropout_rate)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(hidden_size, num_heads, dropout_rate) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(hidden_size, 4)
        # self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, txt_embed, vis_feat, disc_feat):
        # x = self.embedding(x)
        # x = self.positional_encoding(x)
        # x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, txt_embed, vis_feat, disc_feat)

        # x = self.layer_norm(x)
        # x = F.softmax(self.fc(x), dim=-1)
        x = self.fc(x) # maybe change to ffn
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.encoder_attention = nn.MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = FeedForward(hidden_size, dropout_rate)
        # self.dropout = nn.Dropout(dropout_rate)
        # self.layer_norm1 = nn.LayerNorm(hidden_size)
        # self.layer_norm2 = nn.LayerNorm(hidden_size)
        # self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(self, x, encoder_output):
        attended_self = self.self_attention(x, x, x)
        # attended_self = self.dropout(attended_self)
        # x = self.layer_norm1(attended_self + x)
        x = attended_self
        attended_encoder = self.encoder_attention(x, encoder_output, encoder_output)
        # attended_encoder = self.dropout(attended_encoder)
        # x = self.layer_norm2(attended_encoder + x)
        x2 = x + attended_encoder
        x = self.feed_forward(x2)
        # x = self.dropout(x)
        # x = self.layer_norm3(x + attended_encoder)
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + x)
        return x


# class TransformerEncoder(nn.Module):
#     def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout_rate=0.1):
#         super(TransformerEncoder, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.positional_encoding = PositionalEncoding(hidden_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.encoder_layers = nn.ModuleList(
#             [EncoderLayer(hidden_size, num_heads, dropout_rate) for _ in range(num_layers)]
#         )
#         self.layer_norm = nn.LayerNorm(hidden_size)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.positional_encoding(x)
#         x = self.dropout(x)

#         for layer in self.encoder_layers:
#             x = layer(x)

#         x = self.layer_norm(x)
#         return x


# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
#         super(EncoderLayer, self).__init__()
#         self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
#         self.feed_forward = FeedForward(hidden_size, dropout_rate)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layer_norm1 = nn.LayerNorm(hidden_size)
#         self.layer_norm2 = nn.LayerNorm(hidden_size)

#     def forward(self, x):
#         attended = self.attention(x, x, x)
#         attended = self.dropout(attended)
#         x = self.layer_norm1(attended + x)
#         x = self.feed_forward(x)
#         x = self.dropout(x)
#         x = self.layer_norm2(x + attended)
#         return x





# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
#         super(MultiHeadAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_size = hidden_size // num_heads

#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, hidden_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layer_norm = nn.LayerNorm(hidden_size)

#     def forward(self, query, key, value):
#         batch_size, seq_len, _ = query.size()

#         query = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         key = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         value = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
#         attention_weights = F.softmax(scores, dim=-1)
#         attention_weights = self.dropout(attention_weights)
#         attended_values = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#         x = self.fc(attended_values)
#         x = self.dropout(x)
#         x = self.layer_norm(x + attended_values)
#         return x



# class PositionalEncoding(nn.Module):
#     def __init__(self, hidden_size, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.hidden_size = hidden_size

#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size))
#         pe = torch.zeros(max_len, hidden_size)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x * math.sqrt(self.hidden_size)
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len]
#         return x





# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_size = hidden_size // num_heads

#         self.query = nn.Linear(hidden_size, hidden_size)
#         self.key = nn.Linear(hidden_size, hidden_size)
#         self.value = nn.Linear(hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, hidden_size)

#     def forward(self, query, key, value):
#         batch_size, seq_len, _ = query.size()

#         query = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         key = self.key(key).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
#         value = self.value(value).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

#         scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
#         attention_weights = F.softmax(scores, dim=-1)
#         attended_values = torch.matmul(attention_weights, value).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
#         x = self.fc(attended_values)
#         return x
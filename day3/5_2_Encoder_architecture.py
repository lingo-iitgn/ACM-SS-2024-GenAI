import torch
import math
from torch import nn
import torch.nn.functional as F

# Query : what I'm looking for
# Key : what do I have to offer
# Value : what I actually offer on attention

def scaled_dot_product(q, k, v, mask =None):
    # q,k,v = 30 X 8 X 200 X64
    d_k =q.size()[-1] # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # 30 X 8 X 200 X 200
    if mask is not None:
        scaled += mask # 30 X 8 X 200 X 200
    attention = F.softmax(scaled, dim= -1) # 30 X 8 X 200 X 200
    values = torch.matmul(attention, v) # 30 X 8 X 200 X 64 // 64 is here the embedding of the value tensor
    print("Scaled_dot_product ran successfully")
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads =num_heads # 8
        self.head_dim = d_model//num_heads # 64
        self.qkv_layer = nn.Linear(d_model, 3*d_model) # 512 X 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 X 512 
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 X 200 X 512
        qkv = self.qkv_layer(x) # 30 X 200 X 1536
        qkv = qkv.reshape( batch_size, sequence_length, self.num_heads, 3*self.head_dim) # 30 X 200 X 8 X 192
        qkv = qkv.permute( 0, 2, 1, 3) # 30 X 8 X 200 X 192
        q, k, v = qkv.chunk(3, dim=-1) # each are 30 X 8 X 200 X 64
        values, attention = scaled_dot_product(q, k, v, mask) # attention = 30 X 8 X 200 X 200 || Values = 30 X 8 X 200 X 64
        values = values.reshape( batch_size, sequence_length, self.num_heads*self.head_dim) # 30 X 200 X 512(8*64)
        out = self.linear_layer(values)
        print("MultiHeadAttention ran successfully")
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps = 1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # [512]
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(parameters_shape)) # [512]
        self.beta =nn.Parameter(torch.ones(parameters_shape)) # [512]
    
    def forward(self,inputs): # 30 X 200 X 512
        dims = [-(i+1) for i in range(len(self.parameters_shape))] #[-1]
        mean = inputs.mean(dim=dims, keepdim=True) # 30 X 200 X 1
        var = ((inputs-mean)**2).mean(dim=dims, keepdim=True) # 30 X 200 X 1
        std=(var+self.eps).sqrt() # 30 X 200 X 1
        y=(inputs-mean)/std # 30 X 200 X 512
        out=self.gamma*y + self.beta
        print("LayerNormalization ran successfully")
        return out

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionWiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # 512 X 2048
        self.linear2 = nn.Linear(hidden, d_model) # 2048 x 512
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self, x): # 30 X 200 X 512
        x = self.linear1(x) # 30 X 200 X 2048
        x = self.relu(x) # 30 X 200 X 2048
        x = self.dropout(x) # 30 X 200 X 2048
        x = self.linear2(x) # 30 X 200 X 512
        print("PositionWiseFeedForward ran successfully")
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.dropout1=nn.Dropout(p = drop_prob)
        self.ffn = PositionWiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])
        self.dropout2 = nn.Dropout( p =drop_prob)

    def forward(self, x):
        residual_x = x # 30 X 200 X 512
        x = self.attention(x, mask = None) # 30 X 200 X 512
        x = self.dropout1(x) # 30 X 200 X 512
        x = self.norm1(x + residual_x) # 30 X 200 X 512
        residual_x = x # 30 X 200 X 512
        x = self.ffn(x) # 30 X 200 X 512
        x = self.dropout2(x) # 30 X 200 X 512
        x = self.norm2(x + residual_x) # 30 X 200 X 512
        print("EncoderLayer ran successfully")
        return x # This x is now much more context aware compared to x which we received as input.
    
class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        print("Encoder ran successfully")
        return x

d_model = 512 # Size of every single vector
num_heads = 8 # no. of head
drop_prob = 0.1 # number of neurons will be off during forward + backward pass here it is 10%
batch_size = 30 # no. of sequence in each batch
max_seq_len = 200 # max. length of sequence allowed
ffn_hidden = 2048 # Feed Forward layer ( to expand number of neurons from 512 to 2048 and back to 512)
num_layers = 6 # no. of encoder layers to be repeated

encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)  

x = torch.randn((batch_size, max_seq_len, d_model)) # includes the positional encoding
out = encoder(x)
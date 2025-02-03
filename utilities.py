
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096):
        """
        Initialize Rotary Position Embedding
        
        Args:
            dim: Dimension of the embedding (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError("Dimension must be divisible by 2")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position indices tensor
        position = torch.arange(max_seq_len).unsqueeze(1)  # [max_seq_len, 1]
        
        # Create dimension indices tensor for half the dimension
        # Since we'll rotate half the dimensions, we only need dim/2
        div_term = torch.exp(
            torch.arange(0, dim//2) * -(math.log(10000.0) / (dim//2))
        )
        
        # Compute sin and cos tables for half dimensions
        emb = position * div_term
        self.register_buffer("sin_table", emb.sin().unsqueeze(0))  # [1, max_seq_len, dim//2]
        self.register_buffer("cos_table", emb.cos().unsqueeze(0))  # [1, max_seq_len, dim//2]
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embedding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor with positional information encoded
        """
        batch_size, num_heads, seq_len, dim = x.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}")
            
        # Get sin and cos values for current sequence length
        sin = self.sin_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        cos = self.cos_table[:, :seq_len, :]  # [1, seq_len, dim//2]
        
        # Duplicate the sin/cos for the full dimension
        sin = torch.cat([sin, sin], dim=-1)  # [1, seq_len, dim]
        cos = torch.cat([cos, cos], dim=-1)  # [1, seq_len, dim]
        
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1)  # [1, 1, seq_len, dim]
        cos = cos.unsqueeze(1)  # [1, 1, seq_len, dim]
        
        # Expand to match input shape
        sin = sin.expand(batch_size, num_heads, -1, -1)
        cos = cos.expand(batch_size, num_heads, -1, -1)
        
        # Apply rotation using complex number multiplication:
        # (a + ib)(cos θ + i sin θ) = (a cos θ - b sin θ) + i(a sin θ + b cos θ)
        return (x * cos) + (self._rotate_half(x) * sin)
    





class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N,dim = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        #print(q.shape)
        q=self.rope(q)
        k=self.rope(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        # Output projection
        x = self.proj(x)
        return x


##this class represents multi-latent attention, a version of multihead attention that uses low rank shared spaces for k,v for better kv cache memory management
#The original paper stores RoPE of a projection of the key tensor, but this is somewhat dubious to me, whether it is worth the caching
#If it is true that there are no performance downsides to the positional encoding being contained in only half of the embedding dimension, then this is fair game
#however I have not evaluated that
#regardless this is true to the original paper
class MultiLatentAttention(nn.Module):
    def __init__(self,hidden_dim,num_heads=8,low_rank=2):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        #assert hidden_dim//num_heads
        #downproj for q
        self.qd_proj=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.qu_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.qr_proj=nn.Linear(hidden_dim,self.head_dim)
        #shared downproj for k,v
        self.kvd=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.v_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.k_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.kr_proj=nn.Linear(hidden_dim,self.head_dim)
        #output proj
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        self.scale = (2*self.head_dim) ** -0.5

    def forward(self, x):
        #layer norm prior to input
        B, N,dim = x.shape
        
        # query projections
        qd=self.qd_proj(x) #B,N,low_rank_dim
        qr=self.qr_proj(x).unsqueeze(2)# B,N,1,head_dim
        qr=qr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3) #B,num_heads,seq_len,head_dim
        qr=self.rope(qr)
        q=self.qu_proj(qd) #B,N,dim
        q=q.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        q=torch.cat((q,qr),dim=-1) #B,num_heads,seq_len,head_dim*2

        #keys
        low_rank_kv=self.kvd(x)
        k=self.k_up_proj(low_rank_kv)
        kr=self.kr_proj(x).unsqueeze(2)
        kr=kr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3)
        kr=self.rope(kr)
        k= k.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=torch.cat((k,kr),dim=-1) #B,num_heads,seq_len,head_dim*2
        
        #values
        v=self.v_up_proj(low_rank_kv) 
        v=v.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(N, N), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.to(attn.device), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        
        # Output projection
        x = self.o_proj(x)
        return x


#just qkv in case you want to use flash attention 
class MultiLatentQKV(nn.Module):
    def __init__(self,hidden_dim,num_heads=8,low_rank=2):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        #assert hidden_dim//num_heads
        #downproj for q
        self.qd_proj=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.qu_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.qr_proj=nn.Linear(hidden_dim,self.head_dim)
        #shared downproj for k,v
        self.kvd=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.v_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.k_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.kr_proj=nn.Linear(hidden_dim,self.head_dim)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        

    def forward(self, x):
        #layer norm prior to input
        B, N,dim = x.shape
        
        # query projections
        qd=self.qd_proj(x) #B,N,low_rank_dim
        qr=self.qr_proj(x).unsqueeze(2)# B,N,1,head_dim
        qr=qr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3) #B,num_heads,seq_len,head_dim
        qr=self.rope(qr)
        q=self.qu_proj(qd) #B,N,dim
        q=q.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        q=torch.cat((q,qr),dim=-1) #B,num_heads,seq_len,head_dim*2

        #keys
        low_rank_kv=self.kvd(x)
        k=self.k_up_proj(low_rank_kv)
        kr=self.kr_proj(x).unsqueeze(2)
        kr=kr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3)
        kr=self.rope(kr)
        k= k.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=torch.cat((k,kr),dim=-1) #B,num_heads,seq_len,head_dim*2
        
        #values
        v=self.v_up_proj(low_rank_kv) 
        v=v.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        

        return q,k,v
    
#for compiled flex attention we need q,k,v with the same embedding dimension
class AltMultiLatentQKV(nn.Module):
    def __init__(self,hidden_dim,num_heads=8,low_rank=2):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_dim//num_heads
        #assert hidden_dim//num_heads
        #downproj for q
        self.qd_proj=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.qu_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.qr_proj=nn.Linear(hidden_dim,self.head_dim//2)
        #shared downproj for k,v
        self.kvd=nn.Linear(hidden_dim,hidden_dim//low_rank)
        self.v_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.k_up_proj=nn.Linear(hidden_dim//low_rank,hidden_dim)
        self.kr_proj=nn.Linear(hidden_dim,self.head_dim//2)
        self.rope=RotaryPositionEmbedding(self.head_dim)
        

    def forward(self, x):
        #layer norm prior to input
        B, N,dim = x.shape
        
        # query projections
        qd=self.qd_proj(x) #B,N,low_rank_dim
        qr=self.qr_proj(x).unsqueeze(2)# B,N,1,head_dim
        qr=qr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3) #B,num_heads,seq_len,head_dim
        qr=self.rope(qr)
        q=self.qu_proj(qd) #B,N,dim
        q=q.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        q=torch.cat((q,qr),dim=-1) #B,num_heads,seq_len,head_dim*2

        #keys
        low_rank_kv=self.kvd(x)
        k=self.k_up_proj(low_rank_kv)
        kr=self.kr_proj(x).unsqueeze(2)
        kr=kr.expand(-1,-1,self.num_heads,-1).permute(0,2,1,3)
        kr=self.rope(kr)
        k= k.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        k=torch.cat((k,kr),dim=-1) #B,num_heads,seq_len,head_dim*2
        
        #values
        v=self.v_up_proj(low_rank_kv) 
        v=v.reshape(B,N,self.num_heads,self.head_dim).permute(0,2,1,3)
        
        return q,k,v




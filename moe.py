import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

  def __init__(self, hidden_size, ffn_hidden_size):
    super().__init__()
    self.layer=nn.Sequential(
      nn.Linear(hidden_size,ffn_hidden_size),
      nn.GELU(),
      nn.Linear(ffn_hidden_size,hidden_size)
    )

  def forward(self, x):
    # x: (B, S, H)
    return self.layer(x)


#single linear MoE layer, no regularization, just topk=1
class MoE(nn.Module):

  def __init__(self, num_experts, hidden_size, ffn_hidden_size):
    super().__init__()

    

    self.num_experts = num_experts
    self.router = nn.Linear(hidden_size, num_experts)
    self.experts = nn.ModuleList([MLP(hidden_size, ffn_hidden_size) for _ in range(num_experts)])



  def forward(self, x):
    # x: (B, S, H)

    scores = self.router(x) # (B, S, E)
    probs_0 = F.softmax(scores, dim=2) # (B, S, E)
    p,expert_id=torch.max(probs_0,dim=-1)


    out=torch.zeros_like(x)

    for i, expert in enumerate(self.experts):
      mask=expert_id==i
      out[mask,:]=expert(x[mask])
      print(out.shape,p.shape)

    #output=p.unsqueeze(-1)*out
    output=out #for topk=1

    return output
  

#topk sigmoid moe layer, no entropy regularization or loss output
#remember to add all_gather and scatter operations for expert parallelism in implementations that need that
class sigMoE(nn.Module):
  def __init__(self, num_experts, hidden_size, ffn_hidden_size,k=1):
    super().__init__()

    
    self.k=k
    self.num_experts = num_experts
    self.router = nn.Linear(hidden_size, num_experts)
    self.experts = nn.ModuleList([MLP(hidden_size, ffn_hidden_size) for _ in range(num_experts)])
    


  def forward(self, x):
    # x: (B, S, H)
    B,S,H=x.shape
    scores = self.router(x) # (B, S, E)
    probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
    p,expert_id=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
    p=p.unsqueeze(-1) #B,S,K,1
    out=torch.zeros((B,S,self.k,H))
    
    
    for i, expert in enumerate(self.experts):
      #mask=(expert_id==i) #3 dimensional slice (B,S,k)
      B,S,K=torch.where(expert_id==i)
      #print(x[B,S,:].shape)
      
      out[B,S,K,:]=expert(x[B,S,:])
      #print(out.shape,p.shape)
    p=p/p.sum(dim=2)
    output=(p*out).sum(dim=2)
    return output
  
#x = torch.randn((2, 3, 5))
#moe = sigMoE(4, 5, 20,k=2)
#print(x)
#print(moe(x))
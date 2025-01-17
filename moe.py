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


#single linear MoE layer, no regularization
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

    output=p.unsqueeze(-1)*out

    return output
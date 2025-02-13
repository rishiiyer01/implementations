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
  

#topk softmax moe layer, no entropy regularization or loss output
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
  


  




#sparse topk moe with expert parallel for mlp replacement
#took out if else, torch.distributed knows when it doesn't need to gather scatter
class ffMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        self.rank=torch.distributed.get_rank()
        
        #self.experts = nn.ModuleList([FeedForward(hidden_size) for _ in range(num_experts)]) #deprecated feature
        
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([MLP(hidden_size) for _ in range(experts_per_rank)])



        #load balancing
        self.gamma=0.001 #from deepseek v3
        self.register_buffer('b', torch.zeros(num_experts))
        self.register_buffer('uniform', torch.ones(num_experts)/num_experts)
        

    def load_balancing(self,s):
        #auxiliary free load balancing
        #it is pretty annoying to use a loss, so deepseeks method is pretty nice
        meanscores=s.mean(dim=[0,1])
        diff=meanscores-self.uniform #if diff>0 then we subtract if diff<0 then we add 
        s=s+self.b #b broadcasted
        #update bias
        self.b=self.b-(torch.sign(diff)*self.gamma)
        torch.distributed.all_reduce(self.b, op=torch.distributed.ReduceOp.SUM)
        self.b /= self.world_size
        _,expert_id=torch.topk(s,self.k,dim=-1) 
       
        return expert_id

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        expert_id=self.load_balancing(probs_0)
        p,_=torch.topk(probs_0,self.k,dim=-1) #(B,S,K) #you could use this topk for expert routing, but we need aux free load balancing
        p=p.unsqueeze(-1) #B,S,K,1
        #out=torch.empty((b,s,self.k,h))
        #global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2)) #p normalization 
        output=(p*output_local).sum(dim=2)


        return output
    





#x = torch.randn((2, 3, 5))
#moe = sigMoE(4, 5, 20,k=2)
#print(x)
#print(moe(x))




class voMoE(nn.Module):
    def __init__(self, num_experts, hidden_size,k=2):
        super().__init__()
    
        self.rank=torch.distributed.get_rank()
        self.k=k
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_size, num_experts)
        
        self.world_size=torch.distributed.get_world_size()
        #self.experts = nn.ModuleList([nn.Linear(hidden_size,hidden_size) for _ in range(num_experts)]) #deprecated feature
        experts_per_rank = num_experts // self.world_size
        assert num_experts % self.world_size == 0, "Number of experts must be divisible by world size"
        start_idx = self.rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        self.local_experts_ids = list(range(start_idx, end_idx))
        self.local_experts = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(experts_per_rank)])


        #load balancing
        self.gamma=0.001 #from deepseek v3
        self.register_buffer('b', torch.zeros(num_experts))
        self.register_buffer('uniform', torch.ones(num_experts)/num_experts)
        

    def load_balancing(self,s):
        #auxiliary free load balancing
        #it is pretty annoying to use a loss, so deepseeks method is pretty nice
        meanscores=s.mean(dim=[0,1])
        diff=meanscores-self.uniform #if diff>0 then we subtract if diff<0 then we add 
        s=s+self.b #b broadcasted
        #update bias
        self.b=self.b-(torch.sign(diff)*self.gamma)
        torch.distributed.all_reduce(self.b, op=torch.distributed.ReduceOp.SUM)
        self.b /= self.world_size
        _,expert_id=torch.topk(s,self.k,dim=-1) 
       
        return expert_id

    def forward(self, x):
        # x: (B, S, H)
        device=torch.cuda.current_device()
        b,s,h=x.shape
        scores = self.router(x) # (B, S, E)
        probs_0 = F.softmax(scores,dim=-1) # (B, S, E)
        expert_id=self.load_balancing(probs_0)
        p,_=torch.topk(probs_0,self.k,dim=-1) #(B,S,K)
        p=p.unsqueeze(-1) #B,S,K,1
        #out=torch.empty((b,s,self.k,h))
        #global_out=torch.empty((self.world_size*b,s,self.k,h),device=device,dtype=x.dtype) # each gpu will have it's data parallel we need to grab
        global_expert_id=torch.empty((self.world_size*b,s,self.k),device=device,dtype=expert_id.dtype)
        torch.distributed.all_gather_into_tensor(global_expert_id,expert_id) 
        global_x=torch.empty((self.world_size*b,s,h),device=device,dtype=x.dtype) #gathers x from dp
        torch.distributed.all_gather_into_tensor(global_x,x)
        output_total = torch.zeros((self.world_size*b,s,self.k,h),dtype=x.dtype,device=device)
        for i, expert in enumerate(self.local_experts):
            local_expert_id = self.local_experts_ids[i] #1,2,3,4 | 5,6,7,8 for default case
            B, S, K = torch.where(global_expert_id == local_expert_id)
            out = expert(global_x[B, S, :])
            output_total[B, S, K,:] = out #should only store the first 4 k slices on the zeroth gpu
                
        output_local = torch.empty((b, s,self.k, h), device=device,dtype=output_total.dtype)
        #output_local_list=[torch.empty((b,s,h)) for _ in range(self.world_size)]
        #output_total_list=list(torch.tensor_split(output_total,self.world_size,dim=0))
        torch.distributed.reduce_scatter_tensor(output_local, output_total) 
        p=p/(p.sum(dim=2).unsqueeze(2))
        output=(p*output_local).sum(dim=2)
        return output
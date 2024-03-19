import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
### all implementation from https://arxiv.org/pdf/2001.03750.pdf SympNets

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class LASymplecticBlock(nn.Module):

    def __init__(self, dim, channels: int) -> None:
        #### dim is the size of the space of the input space 2*D D = param space
        super(LASymplecticBlock, self).__init__()
        assert (dim % 2) == 0
        assert (channels % 2) == 0
        self.dim = dim
        self.param_dim = dim // 2

        self.bias = nn.Parameter(torch.ones(self.dim))
        self.channels = channels
        self.A = [nn.Linear(self.param_dim,self.param_dim, bias = False) for _ in range(self.channels)]
        for layer in self.A:
            parametrize.register_parametrization(layer, "weight", parametrization=Symmetric())
    
    def forward(self, z) -> torch.Tensor:
        ### assume the first block is up, the second is down and they alternate
        mode = "up"
        
        final_result = torch.matmul(z,torch.eye(self.dim))
        for layer in self.A:
            q, p = torch.hsplit(final_result, (self.dim, 2), -1)
            if mode == "up":
                final_result = torch.cat([q + layer(p), p], -1)
                mode = "down"
            elif mode == "down":
                final_result = torch.cat([q, p + layer(q)], -1)
                mode = "up" ### alternate modes
        return final_result + self.bias

class SympActivation(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super(SympActivation, self).__init__()
        assert (dim % 2) == 0
        self.dim = dim
        self.mode = mode
        self.param_dim = dim // 2
        self.activation = nn.Sigmoid()
        self.a = nn.Parameter(torch.ones(self.param_dim))
    
    def forward(self, z) -> torch.Tensor:
        q, p = torch.hsplit(z, (self.dim, 2), -1)
        if self.mode == "up":
            return torch.cat([q, self.activation(p) * self.a + p], -1)
        elif self.mode == "down":
            return torch.cat([p, self.activation(q)*self.a + q], -1)

        else:
            return z


    

        
class GSymplecticBlock(nn.Module):
    def __init__(self, dim, mode: str) -> None:
        assert (dim % 2) == 0
        assert (mode in ["low", "up"])
        super(GSymplecticBlock, self).__init__()
        self.param_dim = dim // 2
        self.dim = dim
        self.mode = mode


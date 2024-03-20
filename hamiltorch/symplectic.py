import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
### all implementation from https://arxiv.org/pdf/2001.03750.pdf SympNets

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class SymplecticLinearBlock(nn.Module):

    def __init__(self, dim, channels: int) -> None:
        #### dim is the size of the space of the input space 2*D D = param space
        super(SymplecticLinearBlock, self).__init__()
        assert (dim % 2) == 0
        assert (channels % 2) == 0
        self.dim = dim
        self.param_dim = dim // 2

        self.channels = channels

        self.bias = nn.ParameterList([nn.Parameter(torch.ones(self.dim)) for _ in range(self.channels)])
        self.A = nn.ModuleList([nn.Linear(self.param_dim,self.param_dim, bias = False) for _ in range(self.channels)])
        for layer in self.A:
            parametrize.register_parametrization(layer, "weight", parametrization=Symmetric())
            nn.init.orthogonal_(layer.weight)
        for bias in self.bias:
            nn.init.normal_(bias)
    def forward(self, z, dt) -> torch.Tensor:
        ### assume the first block is up, the second is down and they alternate
        #### dt is the step size 
        mode = "up"
        
        final_result = torch.matmul(z,torch.eye(self.dim))
        for bias, layer in zip(self.bias,self.A):
            q, p = torch.hsplit(final_result, 2)
            if mode == "up":
                final_result = torch.cat([q + layer(p)*dt, p], -1) + bias*dt
                mode = "down"
            elif mode == "down":
                final_result = torch.cat([q, p + layer(q)*dt], -1) + bias * dt
                mode = "up" ### alternate modes
        return final_result 

class SymplecticActivation(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super(SymplecticActivation, self).__init__()
        assert (dim % 2) == 0
        assert (mode in ["up", "down"])
        self.dim = dim
        self.mode = mode
        self.param_dim = dim // 2
        self.activation = nn.SiLU()
        self.a = nn.Parameter(torch.ones(self.param_dim))
        nn.init.normal_(self.a)
    def forward(self, z, dt) -> torch.Tensor:
        q, p = torch.hsplit(z, 2)
        if self.mode == "up":
            return torch.cat([q, dt*self.activation(q) * self.a + p], -1)
        elif self.mode == "down":
            return torch.cat([dt*self.activation(p)*self.a + q, p], -1)

        else:
            return z


class LASymplecticBlock(nn.Module):
    def __init__(self, dim, activation_mode: str = "up", channels:int = 4) -> None:
        super(LASymplecticBlock, self).__init__()
        self.linear_block = SymplecticLinearBlock(dim = dim, channels=channels)
        self.activation_block = SymplecticActivation(dim = dim, mode = activation_mode)

    def forward(self, z, dt) -> torch.Tensor:
        return self.activation_block(self.linear_block(z, dt), dt)


class SymplecticNeuralNetwork(nn.Module):
    def __init__(self, dim, activation_modes: list[str], channels: list[int]) -> None:
        super(SymplecticNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([LASymplecticBlock(dim, activation_mode, channel) for (activation_mode, channel) in zip (activation_modes, channels)])
    def step(self, z, dt) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, dt)
        return z

    def forward(self, z, t):
        ### here z is the initial position, and t is a tensor of variable times
        ### this predicts at each of the variable time
        
        # forward_t = lambda t: self.step(z, t)
        # preds = torch.vmap(forward_t, out_dims=0)(t)
        dts = torch.diff(t, prepend = torch.zeros(1))
        preds = []
        for dt in dts:
            z = self.step(z, dt)
            preds.append(z)
        return t, torch.stack(preds,axis=0)



    

        



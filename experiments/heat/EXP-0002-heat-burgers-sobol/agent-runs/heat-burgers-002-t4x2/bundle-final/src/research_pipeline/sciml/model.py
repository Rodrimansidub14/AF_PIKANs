"""Fixed Gaussian RBF KAN, 11 bases + SiLU per edge; physical input derivatives."""
import torch
from torch import nn


class RBFLayer(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.register_buffer('centers',torch.linspace(-1.05,1.05,11))
        self.register_buffer('width',torch.tensor(.21))
        self.coefficients=nn.Parameter(torch.empty(outputs,inputs,11))
        self.base=nn.Parameter(torch.empty(outputs,inputs))
        nn.init.normal_(self.coefficients,std=.1/(inputs**.5))
        nn.init.normal_(self.base,std=.1/(inputs**.5))

    def basis(self,x):
        return torch.exp(-((x.unsqueeze(-1)-self.centers)/self.width)**2)

    def forward(self,x):
        return torch.einsum('big,oig->bo',self.basis(x),self.coefficients)+torch.nn.functional.linear(torch.nn.functional.silu(x),self.base)


class RBFPIKAN(nn.Module):
    def __init__(self,problem):
        super().__init__()
        if problem not in ('heat','burgers'): raise ValueError('unknown problem')
        self.problem=problem
        self.layers=nn.ModuleList([RBFLayer(a,b) for a,b in zip([2,16,16],[16,16,1])])

    def normalized(self,x):
        return torch.cat([2*x[:,0:1]-1 if self.problem=='heat' else x[:,0:1],2*x[:,1:2]-1],dim=1)

    def forward(self,x):
        # PINA's LabelTensor is unwrapped without detaching its autograd graph.
        if hasattr(x,'tensor'): x=x.tensor
        h=self.normalized(x)
        for i,layer in enumerate(self.layers):
            h=layer(h)
            if i<2: h=torch.tanh(h)
        if self.problem=='heat':
            return torch.sin(torch.pi*x[:,0:1])+x[:,1:2]*x[:,0:1]*(1-x[:,0:1])*h
        return h


def derivatives(y,x):
    gradient=torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True,retain_graph=True)[0]
    ux=gradient[:,0:1]; ut=gradient[:,1:2]
    uxx=torch.autograd.grad(ux,x,torch.ones_like(ux),create_graph=True,retain_graph=True)[0][:,0:1]
    return ux,ut,uxx

import torch
import math


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.mean(dim=1, keepdim=True)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_in = grad_output.expand(input.shape) / float(input.shape[1])
        return grad_in

class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus = SegmentConsensus.apply

    def forward(self, input):
        return self.consensus(input)

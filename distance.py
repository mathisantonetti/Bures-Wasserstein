import torch
from torch.autograd import Function
import numpy as np
import scipy
import ot
import time

# Piece of code taken from https://github.com/steveli/pytorch-sqrtm (it is suboptimal if we use a gpu but we can avoid its use by modelling the square root of Sigma instead of Sigma)
class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

def manual_einsum(string, t1, t2):
    return torch.tensor(np.einsum(string, t1.detach().cpu().numpy(), t2.detach().cpu().numpy()))

def tracesquared(A, dim=0):
    res = 0
    for i in range(A.shape[-1]):
        res += torch.sum(A[...,i,:]*A[...,:,i], dim=dim)

    return res

def batch_trace(t):
    return t.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

def bures_wasserstein_v1(mu1, mu2, sig1, sig2, is_squared=False):
    if(is_squared):
        trace_sq = torch.trace(sig1+sig2)
        sig1_12, sig2_12 = sqrtm(sig1) @ sqrtm(sig2)
        return trace_sq-2*torch.trace(sqrtm((sig1_12.T)@sig2_12))+torch.sum((mu1-mu2)**2)
    else:
        sig21 = sig2@sig1
        return torch.trace(sig1@sig1+sig2@sig2)-2*torch.trace(sqrtm((sig21.T)@sig21))+torch.sum((mu1-mu2)**2)
    
def bures_wasserstein_v2(mu1, mu2, sig1, sig2, is_squared=False):
    sig21 = sig2@sig1
    return torch.einsum('ij,ji->', sig1, sig1) + torch.einsum('ij,ji->', sig2, sig2) - 2*torch.trace(sqrtm((sig21.T)@sig21)) + torch.sum((mu1-mu2)**2)

# This version breaks the gradient in the current implementation but it allows to see the problem with einsum
def bures_wasserstein_v3(mu1, mu2, sig1, sig2, is_squared=False):
    sig21 = sig2@sig1
    return manual_einsum('ij,ji->', sig1, sig1) + manual_einsum('ij,ji->', sig2, sig2) - 2*torch.trace(sqrtm((sig21.T)@sig21)) + torch.sum((mu1-mu2)**2)

def bures_wasserstein_v4(mu1, mu2, sig1, sig2, is_squared=False):
    last_dim = len(mu1.shape)-1
    return batch_trace(sig1@sig1+sig2@sig2) - 2*torch.sum(torch.linalg.svdvals(sig2@sig1), dim=last_dim-1) + torch.sum((mu1-mu2)**2, dim=last_dim)

def bures_wasserstein_v5(mu1, mu2, sig1, sig2, is_squared=False):
    last_dim = len(mu1.shape)-1
    return tracesquared(sig1)+tracesquared(sig1) - 2*torch.sum(torch.linalg.svdvals(sig2@sig1), dim=last_dim-1) + torch.sum((mu1-mu2)**2, dim=last_dim)
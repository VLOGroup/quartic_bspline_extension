import torch
from typing import Tuple, Any

from bspline_cuda import quartic_bspline_forward

class QuarticBSplineFunction(torch.autograd.Function):

    @staticmethod
    def forward(x: torch.Tensor, weights: torch.Tensor, centers: torch.Tensor, scale: float) -> Tuple[torch.Tensor, ...]:
        y, y_prime = quartic_bspline_forward(x, weights, centers, scale)
        return y, y_prime

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, outputs: Tuple):
        x, weights, *_ = inputs
        y, y_prime = outputs

        ctx.save_for_backward(x, weights, y, y_prime)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: Any):

        x, weights, y, y_prime = ctx.saved_tensors

        return None
import torch
from typing import Tuple, Optional, Any

from . import quartic_bspline_forward, quartic_bspline_backward

class QuarticBSplineFunction(torch.autograd.Function):
    """
    Class implementing custom autograd function based on the CUDA kernels
        * quartic_bspline_forward_function()
        * quartic_bspline_backward_function()
    """

    @staticmethod
    def forward(x: torch.Tensor, weights: torch.Tensor, centers: torch.Tensor, scale: float) -> Tuple[torch.Tensor, ...]:
        y, y_prime = quartic_bspline_forward(x, weights, centers, scale)
        return y, y_prime

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, outputs: Tuple) -> None:
        x, weights, centers, scale = inputs
        _, y_prime = outputs

        ctx.save_for_backward(x, weights, centers, y_prime)
        ctx.scale = scale

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: torch.Tensor) -> Tuple:

        x, weights, centers, y_prime = ctx.saved_tensors
        scale = ctx.scale

        grad_x = y_prime * grad_outputs[0]
        grad_w = quartic_bspline_backward(x, weights, centers, scale, grad_outputs[0])[0]

        return grad_x, grad_w, None, None
    
    @staticmethod
    def vmap(
        info: Any, 
        in_dims: Tuple[Optional[int]], 
        x: torch.Tensor, 
        weights: torch.Tensor, 
        centers: torch.Tensor, 
        scale: float
    ) -> Tuple:
        # NOTE
        #   > Since forward function handles already batch-wise tensor evaluation, 
        #       one needs only to make sure that forward call is applied to 
        #       contiguous tensors.
        #   > vmap needs to handle only forward call - backward call does not
        #       need to be considered.
        #   > For signature of input/output of this function see
        #       https://docs.pytorch.org/docs/main/notes/extending.func.html

        return QuarticBSplineFunction.apply(x.contiguous(), weights, centers, scale), (0, 0)

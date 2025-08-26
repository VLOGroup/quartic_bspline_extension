from typing import Any, Tuple
import torch

def forward_func(x_scaled, coeffs, weights):
    y = (coeffs[..., 0]
         + x_scaled * (coeffs[..., 1]
                       + x_scaled * (coeffs[..., 2] +
                                     x_scaled * (coeffs[..., 3] + x_scaled * coeffs[..., 4]))))
    return y * weights.view(1, 1, -1, 1, 1)

def backward_func(x_scaled, b, c, d, e, weights):
    y = (b + x_scaled * (2 * c + x_scaled * (3 * d + x_scaled * 4 * e)))
    return y * weights.view(1, 1, -1, 1, 1)

class QuarticBSplineFunction(torch.autograd.Function):

    @staticmethod
    def forward(x_scaled, scale, coeffs, weights) -> Any:
        return forward_func(x_scaled, coeffs, weights), coeffs[..., 1], coeffs[..., 2], coeffs[..., 3], coeffs[..., 4]

    @staticmethod
    def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs: Tuple, outputs: Tuple):
        x_scaled, scale, _, weights = inputs
        _, b, c, d, e = outputs
        ctx.save_for_backward(x_scaled, weights, b, c, d, e)
        ctx.scale = scale

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grad_outputs: Any) -> Any:
        x_scaled, weights, b, c, d, e = ctx.saved_tensors
        scale = ctx.scale
        y = backward_func(x_scaled, b, c, d, e, weights / scale)
        return grad_outputs[0] * y, *[None] * 3

class QuarticBSplinePotential(torch.nn.Module):

    SUPP_LOWER = -2.5
    SUPP_UPPER = 2.5
    NUM_NODES = 6

    def __init__(self):
        super().__init__()

        coeffs = torch.zeros(7, 5)
        coeffs[1, :] = torch.tensor([625 / 16, 125 / 2, 75 / 2, 10.0, 1.0]) / 24
        coeffs[2, :] = torch.tensor([55 / 4, -5.0, -30.0, -20.0, -4.0]) / 24.0
        coeffs[3, :] = torch.tensor([115 / 8, 0.0, -15.0, 0.0, 6]) / 24.0
        coeffs[4, :] = torch.tensor([55 / 4, 5.0, -30.0, 20.0, -4.0]) / 24.0
        coeffs[5, :] = torch.tensor([625 / 16, -125 / 2, 75 / 2, -10.0, 1.0]) / 24
        self.register_buffer('coeffs', coeffs)

        box_lower = -3
        box_upper = 3
        num_centers = 33

        centers = torch.linspace(box_lower, box_upper, num_centers)
        self.register_buffer('centers', centers)
        self.weights = torch.nn.Parameter(torch.log(1 + centers ** 2), requires_grad=True)

        self.scale = 1 # (box_upper - box_lower) / (num_centers - 1)

    @classmethod
    def _bucketise(cls, x):
        x_scaled = (x - cls.SUPP_LOWER) / (cls.SUPP_UPPER - cls.SUPP_LOWER)
        return torch.clamp((x_scaled * (cls.NUM_NODES - 1)).ceil().long(), min=0, max=cls.NUM_NODES)

    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        x_scaled = x.unsqueeze(dim=2)
        x_scaled = (x_scaled - self.centers.view(1, 1, -1, 1, 1)) / self.scale
        index_tensor = self._bucketise(x_scaled)

        coeffs = self.coeffs[index_tensor]
        y, *_ = QuarticBSplineFunction.apply(x_scaled, self.scale, coeffs, self.weights)

        return torch.sum(y) if reduce else torch.sum(y, dim=(2))
import torch

def forward_func(x_scaled, coeffs, weights):
    y = (coeffs[..., 0]
         + x_scaled * (coeffs[..., 1]
                       + x_scaled * (coeffs[..., 2] +
                                     x_scaled * (coeffs[..., 3] + x_scaled * coeffs[..., 4]))))
    return y * weights.view(1, 1, -1, 1, 1)

class QuarticBSplinePotential(torch.nn.Module):

    SUPP_LOWER = -2.5
    SUPP_UPPER = 2.5
    NUM_NODES = 6

    def __init__(self) -> None:
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

        self.scale = (box_upper - box_lower) / (num_centers - 1)

    @classmethod
    def _bucketise(cls, x) -> torch.Tensor:
        x_scaled = (x - cls.SUPP_LOWER) / (cls.SUPP_UPPER - cls.SUPP_LOWER)
        return torch.clamp((x_scaled * (cls.NUM_NODES - 1)).ceil().long(), min=0, max=cls.NUM_NODES)

    def forward(self, x: torch.Tensor, reduce: bool=True) -> torch.Tensor:
        x_scaled = x.unsqueeze(dim=2)
        x_scaled = (x_scaled - self.centers.view(1, 1, -1, 1, 1)) / self.scale
        index_tensor = self._bucketise(x_scaled)

        coeffs = self.coeffs[index_tensor]

        y = forward_func(x_scaled, coeffs, self.weights)

        return torch.sum(y) if reduce else torch.sum(y, dim=(2))
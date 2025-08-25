import torch
from tests.QuarticBSplinePotential import QuarticBSplinePotential
from bspline_cuda import quartic_bspline_forward

def test_forward():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.rand(10, 48, 25, 25, device=device, requires_grad=True).to(device=device)
    pot = QuarticBSplinePotential().to(device=device)

    y_true = pot(x)
    y_test = quartic_bspline_forward()

    print('wuahaha')


import torch
from matplotlib import pyplot as plt
import pytest

from tests.QuarticBSplinePotential import QuarticBSplinePotential
from bspline_cuda.functions import QuarticBSplineFunction

@pytest.fixture(autouse=True)
def seed_random_number_generators():
    seed_val = 123
    torch.manual_seed(seed_val)

def test_forward():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # NOTE
    #   > For testing choose dtype = torch.float64
    #   > For printing tensors with high precision use 
    #       torch.set_printoptions(precision=8) 

    bs = 10
    f = 48
    w = 64
    h = 128

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype) - 1) 
    
    pot = QuarticBSplinePotential().to(device=device, dtype=dtype)
    y_true = pot(x, reduce=False)

    weight_tensor = torch.cat([pot.weights.reshape(1, -1) for _ in range(0, f)], dim=0)
    centers = pot.centers
    scale = pot.scale

    y_test, _ = QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)

    assert torch.allclose(y_true, y_test)

def test_backward():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    bs = 10
    f = 4
    w = 128
    h = 64

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype) - 1)
    x.requires_grad_(True)
    pot = QuarticBSplinePotential().to(device=device, dtype=dtype)

    weight_tensor = torch.cat([pot.weights.reshape(1, -1) for _ in range(0, f)], dim=0)
    centers = pot.centers
    scale = pot.scale

    with torch.enable_grad():
        y_true = pot(x)
        y_test, _ = QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)

    dy_dx_true = torch.autograd.grad(inputs=x, outputs=y_true, retain_graph=True)[0]
    dy_dw_true = torch.autograd.grad(inputs=[p for p in pot.parameters() if p.requires_grad], outputs=y_true)[0]

    dy_dx_test = torch.autograd.grad(inputs=x, outputs=torch.sum(y_test), retain_graph=True)[0]
    dy_dw_test = torch.autograd.grad(inputs=[p for p in pot.parameters() if p.requires_grad], outputs=torch.sum(y_test))[0]

    assert torch.allclose(dy_dx_true, dy_dx_test) and torch.allclose(dy_dw_true, dy_dw_test)

def test_tensors_on_different_devices():
    dtype = torch.float64

    device_1 = torch.device('cuda:0')
    device_2 = torch.device('cpu')

    bs = 10
    f = 4
    w = 64
    h = 32
    num_centers = 33

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device_1, dtype=dtype) - 1)
    weight_tensor = torch.rand(f, num_centers, device=device_1, dtype=dtype)
    box_lower = -3
    box_upper = 3
    centers = torch.linspace(box_lower, box_upper, num_centers, 
                             device=device_2, dtype=dtype)
    scale = (box_upper - box_lower) / (num_centers - 1)

    expected_err_msg = 'Tensors must be on the same device.'
    err_msg = ''
    try:
        y, _ = QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)
    except RuntimeError as e:
        err_msg = str(e)
    finally:
        assert expected_err_msg == err_msg

def test_tensors_of_different_datatype():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype_1 = torch.float64
    dtype_2 = torch.float32

    bs = 10
    f = 4
    w = 32
    h = 32
    num_centers = 33

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype_1) - 1)
    weight_tensor = torch.rand(f, num_centers, device=device, dtype=dtype_2)
    box_lower = -3
    box_upper = 3
    centers = torch.linspace(box_lower, box_upper, num_centers, 
                             device=device, dtype=dtype_1)
    scale = (box_upper - box_lower) / (num_centers - 1)

    expected_err_msg = 'Tensors must have the same data type.'
    err_msg = ''
    try:
        y, _ = QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)
    except RuntimeError as e:
        err_msg = str(e)
    finally:
        assert expected_err_msg == err_msg





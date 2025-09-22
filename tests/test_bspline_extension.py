import torch
import pytest

from tests.QuarticBSplinePotential import QuarticBSplinePotential
from quartic_bspline_extension.functions import QuarticBSplineFunction

dtypes = [torch.float64]
devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda:0'))

@pytest.fixture(autouse=True)
def seed_random_number_generators():
    seed_val = 123
    torch.manual_seed(seed_val)

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', dtypes)
def test_forward(device: torch.device, dtype: torch.dtype) -> None:
    # NOTE
    #   > For testing choose dtype = torch.float64; for other types torch.allclose()
    #       with default tolerance may be too strict
    #   > For printing tensors with high precision use 
    #       torch.set_printoptions(precision=8) 

    bs = 10
    f = 4
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

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', dtypes)
def test_backward(device: torch.device, dtype: torch.dtype) -> None:
    bs = 10
    f = 2
    w = 64
    h = 128

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

@pytest.mark.parametrize('dtype', dtypes)
def _test_tensors_on_different_devices(dtype: torch.dtype) -> None:
    device_1 = torch.device('cuda:0')
    device_2 = torch.device('cpu')

    bs = 10
    f = 4
    w = 64
    h = 32
    x = 5 * (2 * torch.rand(bs, f, w, h, device=device_1, dtype=dtype) - 1)

    num_centers = 33
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

@pytest.mark.parametrize('device', devices)
def test_tensors_of_different_datatype(device: torch.device) -> None:
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

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', dtypes)
def test_vmap_forward(device: torch.device, dtype: torch.dtype):

    bs = 10
    f = 4
    w = 128
    h = 64
    num_centers = 33

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype) - 1)
    x.requires_grad_(True)

    weight_tensor = torch.rand(f, num_centers, device=device, dtype=dtype)
    box_lower = -3
    box_upper = 3
    centers = torch.linspace(box_lower, box_upper, num_centers, 
                             device=device, dtype=dtype)
    scale = (box_upper - box_lower) / (num_centers - 1)

    def per_sample_func(x_: torch.Tensor) -> torch.Tensor:
        y, _ = QuarticBSplineFunction.apply(x_, weight_tensor, centers, scale)
        return y

    def func(*x_: torch.Tensor) -> torch.Tensor:
        return torch.vmap(per_sample_func)(torch.cat(x_, dim=0))   

    assert torch.allclose(func(*[x]), QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)[0])

@pytest.mark.parametrize('device', devices)
@pytest.mark.parametrize('dtype', dtypes)
def test_vmap_backward(device: torch.device, dtype: torch.dtype):

    bs = 10
    f = 4
    w = 128
    h = 64
    num_centers = 33

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype) - 1)
    x.requires_grad_(True)

    weight_tensor = torch.rand(f, num_centers, device=device, dtype=dtype)
    box_lower = -3
    box_upper = 3
    centers = torch.linspace(box_lower, box_upper, num_centers, 
                             device=device, dtype=dtype)
    scale = (box_upper - box_lower) / (num_centers - 1)

    def per_sample_func(x_: torch.Tensor) -> torch.Tensor:
        y, _ = QuarticBSplineFunction.apply(x_, weight_tensor, centers, scale)
        return y

    def func(*x_: torch.Tensor) -> torch.Tensor:
        return torch.vmap(per_sample_func)(torch.cat(x_, dim=0))   

    y_true, _ = QuarticBSplineFunction.apply(x, weight_tensor, centers, scale)
    dy_dx_true = torch.autograd.grad(inputs=x, outputs=torch.sum(y_true))[0]

    with torch.enable_grad():
        y_test = torch.sum(func(*[x]))
    dy_dx_test = torch.autograd.grad(inputs=x, outputs=y_test)[0]

    assert torch.allclose(dy_dx_true, dy_dx_test)

import torch
from matplotlib import pyplot as plt

from tests.QuarticBSplinePotential import QuarticBSplinePotential
from bspline_cuda import quartic_bspline_forward

VISUALISE = False

def test_forward():
    torch.manual_seed(123)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    # NOTE
    #   > For testing choose dtype = torch.float64
    #   > For printing tensors with high precision use 
    #       torch.set_printoptions(precision=8) 

    bs = 5
    f = 48
    w = 56
    h = 56

    x = 5 * (2 * torch.rand(bs, f, w, h, device=device, dtype=dtype) - 1)
    pot = QuarticBSplinePotential().to(device=device, dtype=dtype)
    y_true = pot(x, reduce=False)

    weight_tensor = torch.cat([pot.weights.reshape(1, -1) for _ in range(0, f)], dim=0)
    centers = pot.centers
    scale = pot.scale

    y_test, _ = quartic_bspline_forward(x, weight_tensor, centers, scale)


    if VISUALISE:
        f = 3 
        t = torch.stack([torch.linspace(-2, 2, 111)
                        for _ in range(0, f)]).unsqueeze(dim=1).unsqueeze(dim=0).to(device=device)

        ww = torch.cat([weight_tensor for _ in range(0, f)], dim=0)

        y, _ = quartic_bspline_forward(t, ww, centers, scale)

        fig = plt.figure()
        for i in range(0, f):
            ax = fig.add_subplot(3, 1, 1 + i)
            ax.plot(t[0, 0, 0, :].detach().cpu().numpy(), y[0, i, 0, :].detach().cpu().numpy(), 
                    color='orange')
        plt.show()

    assert torch.allclose(y_true, y_test)

if __name__ == '__main__':
    test_forward()



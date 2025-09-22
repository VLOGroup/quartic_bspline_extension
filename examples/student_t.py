import torch
from matplotlib import pyplot as plt

from quartic_bspline_extension.functions import QuarticBSplineFunction

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    box_lower = -3.0
    box_upper = 3.0
    num_centers = 77
    centers = torch.linspace(box_lower, box_upper, num_centers)
    weights_1 = torch.log(1 + centers ** 2)
    weights_2 = torch.abs(centers)
    weights = torch.stack([weights_1, weights_2], dim=0)
    scale = (box_upper - box_lower) / (num_centers - 1)

    f = 2
    t = torch.stack([torch.linspace(box_lower, box_upper, 111)
                     for _ in range(0, f)]).unsqueeze(dim=1).unsqueeze(dim=0)
    
    centers = centers.to(device=device, dtype=dtype)
    weights = weights.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)
    t.requires_grad_(True)
    
    y, _ = QuarticBSplineFunction.apply(t, weights, centers, scale)

    dy_dt = torch.autograd.grad(inputs=t, outputs=torch.sum(y))[0]

    fig_width = 8
    fig_height = 10
    fig = plt.figure(figsize=(fig_width, fig_height))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.plot(t[0, 0, 0, :].detach().cpu().numpy(), y[0, 0, 0, :].detach().cpu().numpy(), 
              color='blue', label='log student t')
    ax_1.plot(t[0, 1, 0, :].detach().cpu().numpy(), y[0, 1, 0, :].detach().cpu().numpy(), 
            color='orange', label='abs')
    ax_1.set_title('potentials')
    ax_1.legend()

    ax_2 = fig.add_subplot(2, 1, 2)
    ax_2.plot(t[0, 0, 0, :].detach().cpu().numpy(), dy_dt[0, 0, 0, :].detach().cpu().numpy(), 
            color='blue', label='log student t')
    ax_2.plot(t[0, 1, 0, :].detach().cpu().numpy(), dy_dt[0, 1, 0, :].detach().cpu().numpy(), 
            color='orange', label='abs')
    ax_2.set_title('derivatives of potentials')

    plt.show()

if __name__ == '__main__':
    main()
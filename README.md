# B-Spline extension

This package implements quartic (midpoint cardinal) b-spline potential functions. By such 
a potential function we mean a parameter-dependent function $\rho(\cdot, \gamma)$ with

$$\rho(x, \gamma) = \sum_{\nu = 1}^{N}\gamma_{\nu}M_{4}(\frac{x - t_{\nu}}{s})$$,

where $\{t_{\nu}\}_{\nu = 1}^{N}$ is an equidistant partition of the Interval $I = [a, b]$, 
$s > 0$ is a scaling parameter and $M_{4}$ refers to the central quartic midpoint cardinal
b-spline (see [[1]](#1)).

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [References](#references)

## Features

- CUDA kernels for forward and backward step
- Custom autograd functions based on these kernels to compute in particular
    the gradients of the potential w.r.t. to input and parameters.
- Feature-wise implementation: 
    * The potential function is designed for input tensors of shape [bs, f, w, h].
    * For a weight tensor of shape [f, N], the potential w.r.t. to the weights
        [$\nu$, :] is applied to $f$-th channel of the input tensor.

## Installation



## Usage

```python

from bspline_cuda.functions import QuarticBSplineFunction

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

```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## References

<a id="1">[1]</a> 
Schoenberg, Isaac J, 1973.
Cardinal spline interpolation.
SIAM.

## License

MIT License

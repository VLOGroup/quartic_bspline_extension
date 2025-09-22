import torch
import numpy as np
import os
from typing import Callable
import datetime

from quartic_bspline_extension.functions import QuarticBSplineFunction
from Timer import Timer

def profile(func: Callable, x: torch.Tensor, chrome_export: bool=False) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities += [torch.profiler.ProfilerActivity.CUDA]

    profile_memory = False
    record_shapes = False

    skip_first = 3
    wait = 2
    warmup = 2
    active = 2
    num_iterations = skip_first + wait + warmup + active + 3

    schedule = torch.profiler.schedule(skip_first=3, wait=2, warmup=2, active=2)
    with torch.profiler.profile(activities=activities, schedule=schedule,
                                profile_memory=profile_memory, 
                                record_shapes=record_shapes) as prof:
        for _ in range(0, num_iterations):
            _ = func(x)
            prof.step()

    if chrome_export:
        frmt = '%Y_%m_%dT%H_%M_%S%z'
        file_name = '{:s}'.format(datetime.datetime.now().strftime(frmt))
        export_path = os.path.join(os.getcwd(), 'profiling', file_name)
        prof.export_chrome_trace(export_path)

    print(prof.key_averages().table())

def measure_speed_forward(x: torch.Tensor, weights: torch.Tensor, centers: torch.Tensor, 
                          scale: float, device: torch.device, num_timings: int=100) -> None:
    print('profile forward pass ...')

    timings = []
    for _ in range(0, num_timings):
        with Timer(device) as t:
            y, _ = QuarticBSplineFunction.apply(x, weights, centers, scale)
        timings.append(t.time_delta())

    print('median: {:.5f}'.format(np.median(timings)))
    print('mean: {:.5f}'.format(np.mean(timings)))
    print('mean (tail): {:.5f}'.format(np.mean(timings[5::])))

def measure_speed_backward(x: torch.Tensor, weights: torch.Tensor, centers: torch.Tensor, 
                           scale: float, device: torch.device, num_timings: int=100) -> None:
    print('profile backward pass ...')
    
    x.requires_grad_(True)

    timings = []
    for _ in range(0, num_timings):
        with Timer(device) as t:
            y, _ = QuarticBSplineFunction.apply(x, weights, centers, scale)
            _ = torch.autograd.grad(inputs=x, outputs=torch.sum(y))[0]
        timings.append(t.time_delta())

    print('median: {:.5f}'.format(np.median(timings)))
    print('mean: {:.5f}'.format(np.mean(timings)))
    print('mean (tail): {:.5f}'.format(np.mean(timings[5::])))

def forward_func_factory(weights, centers, scale) -> Callable:
    def forward_func(x: torch.Tensor) -> torch.Tensor:
        return QuarticBSplineFunction.apply(x, weights, centers, scale)[0]
    return forward_func

def gradient_func_factory(weights, centers, scale) -> Callable:
    func = forward_func_factory(weights, centers, scale)
    def gradient_func(x: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        y = torch.sum(func(x))
        return torch.autograd.grad(inputs=x, outputs=y)[0]
    return gradient_func
        
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    bs = 128
    f = 48
    w = 64
    h = 64

    box_lower = -2.0
    box_upper = 2.0
    num_centers = 33
    centers = torch.linspace(box_lower, box_upper, 
                             num_centers).to(device=device, dtype=dtype)
    scale = (box_upper - box_lower) / (num_centers - 1)
    weights_ = torch.log(1 + centers ** 2).to(device=device, dtype=dtype)
    weights = torch.stack([weights_ for _ in range(0, f)], dim=0)
    x = 3 * (2 * (torch.rand(bs, f, w, h).to(device=device, dtype=dtype) - 1))

    # ### measure SPEED of forward and backward pass

    measure_speed_forward(x, weights, centers, scale, device)
    measure_speed_backward(x, weights, centers, scale, device)

    # ### PROFILE forward and backward pass
    forward_func = forward_func_factory(weights, centers, scale)
    gradient_func = gradient_func_factory(weights, centers, scale)

    profile(forward_func, x)
    profile(gradient_func, x)

if __name__ == '__main__':
    main()
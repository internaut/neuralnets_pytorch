"""
Using PyTorch auto-grad to compute the derivative of a function.

Defines several functions like polynomials, trigonometric functions and composed functions. All functions are plotted,
the gradients are calculated using PyTorch and are checked against the analytic results. Finally, the derivative
function is plotted.

Nov. 2022

author: Markus Konrad <post@mkonrad.net>
"""

import torch
import matplotlib.pyplot as plt

#%% define the functions

# define some functions that are reused
fn_inner = lambda x: 3.0 * x**2 - 3.0
d_fn_inner = lambda x: 6.0 * x
fn_outer = torch.sin
d_fn_outer = torch.cos

funcs = [  # function as text, derivative as text, function as code, derivative as code
    ('y=2x+1', "y'=2", lambda x: 2.0 * x + 1.0, lambda x: 2.0),
    ('y=3x²-3', "y'=6x", fn_inner, d_fn_inner),
    ('y=4x³-2x²+3x', "y'=12x²-4x+3", lambda x: 4.0 * x**3 - 2.0 * x**2 + 3.0 * x,
                                     lambda x: 12.0 * x**2 - 4.0 * x + 3.0),
    ('y=sin(x)', "y'=cos(x)", fn_outer, d_fn_outer),
    # function composition (chain rule)
    ('y=sin(3x²-3)', "y'=6x cos(3x²-3)", lambda x: fn_outer(fn_inner(x)),
                                         lambda x: d_fn_inner(x) * d_fn_outer(fn_inner(x))),
]

#%%

for title, d_title, fn, d_fn in funcs:
    # evaluate at 100 points in a given range
    xs = torch.linspace(-5, 5, 100)
    ys = fn(xs)

    # collect gradients
    grads = []
    for x in xs:
        x0 = torch.scalar_tensor(x, requires_grad=True)
        y0 = fn(x0)
        y0.backward()

        grads.append(x0.grad.item())

    # check against the analytic result
    if len(set(grads)) == 1:   # constant gradient
        assert grads[0] == d_fn(xs)
    else:
        assert torch.allclose(torch.tensor(grads, dtype=torch.float), d_fn(xs))

    # plot function on top, derivative on bottom
    fig, (ax_upr, ax_lwr) = plt.subplots(2, 1, sharex=True)

    ax_upr.plot(xs, ys)
    ax_upr.set_title(title)
    ax_upr.set_ylabel('y')

    ax_lwr.plot(xs, grads)
    ax_lwr.set_title(d_title)
    ax_lwr.set_xlabel('x')
    ax_lwr.set_ylabel('y')

    fig.show()

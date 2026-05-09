import autograd.numpy as np
import torch
from torchdiffeq import odeint


def hamiltonian(qp):
    q, p = torch.split(qp, 2)
    H = 0.5 * (q**2 + p**2)
    return H


def gradient_hamiltonian(q, p):
    return torch.tensor([q, p])


def state_trajectory(t, qp):
    q = qp[0]
    p = qp[1]
    J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    R = get_R(q, p)
    G = torch.tensor([[0], [1]])
    u = [get_u(t)]
    x_dot = (J-R) @ gradient_hamiltonian(q, p)# + G @ u
    return x_dot


def get_R(q, p, b=1):
    R = torch.tensor(np.array([[0.0, 0.0], [0.0, b * p**2]]))
    return R


def get_u(t):
    return torch.tensor(0.)


def get_trajectory(t_span=[0, 6], timescale=5):
    t_eval = torch.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial value
    x0 = np.random.rand(2) * 2 - 1
    radius = np.random.rand() * 0.9 + 0.1
    x0 = x0 / np.sqrt((x0 ** 2).sum()) * radius
    x0 = torch.tensor(x0)

    spring_ivp = odeint(func=state_trajectory, t=t_eval, y0=x0, rtol=1e-10, method='implicit_midpoint')
    q, p = spring_ivp[:, 0], spring_ivp[:, 1]
    dx = []
    for i in range(timescale * t_span[1]):
        dx.append(state_trajectory(t_eval[i], spring_ivp[i]))
    dx = torch.stack(dx).T

    dq, dp = dx[0], dx[1]

    # If you want to recreate the data used in the thesis, comment this in.
    # Otherwise, remove and use the option in get_dataset
    #q += q * np.random.randn(*q.shape) * 0
    #p += p * np.random.randn(*p.shape) * 0
    return q, p, t_eval, dq, dp


def get_dataset(seed=0, samples=25, t_span=[0, 6], timescale=5, noise=0.):
    data = {'meta': locals()}
    np.random.seed(seed)

    # train data
    xs, dxs, ts = [], [], []
    for s in range(samples):
        q, p, t, dq, dp = get_trajectory(t_span=t_span, timescale=timescale)
        xs.append(torch.stack([q, p]).T)
        dxs.append(torch.stack([dq, dp]).T)
        if s == 0:
            ts = t

    data['x'] = np.concatenate(xs)
    data['dx'] = np.concatenate(dxs).squeeze()
    data['t'] = ts

    # test data
    xs, dxs = [], []
    for s in range(25):
        q, p, t, dq, dp = get_trajectory(t_span=t_span, timescale=timescale)
        xs.append(torch.stack([q, p]).T)
        dxs.append(torch.stack([dq, dp]).T)
        if s == 0:
            ts = t

    data['test_x'] = np.concatenate(xs)
    data['test_dx'] = np.concatenate(dxs).squeeze()
    data['test_t'] = ts

    # This is how noise should be added
    if noise != 0:
        data['x'] += data['x'] * np.random.randn(*data['x'].shape) * noise
    return data

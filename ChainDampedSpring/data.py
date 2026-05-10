import autograd.numpy as np
import torch
from torchdiffeq import odeint
import math


def create_matrices(dimension=3):
    J_small = np.array([[0, 1], [-1, 0]])
    c = 1
    k = 4
    m = 4

    J = np.zeros((2 * dimension, 2 * dimension))
    R = np.zeros((2 * dimension, 2 * dimension))
    Q = np.zeros((2 * dimension, 2 * dimension))
    G = np.zeros((2 * dimension, 2))
    G[1, 0] = 1
    G[3, 1] = 1

    for i in range(dimension):
        J[2*i:2*i + 2, 2*i:2*i + 2] = J_small
        R[2*i+1, 2*i+1] = c

        if i == 0:
            Q[0:2, 0:2] = np.array([[k, 0], [0, 1/m]])
        else:
            Q[2*i:2*i + 2, 2*i:2*i + 2] = np.array([[2*k, 0], [0, 1/m]])

        if i < dimension-1:
            Q[2*i, 2 + 2*i] = -k
            Q[2 + 2*i, 2*i] = -k

    A = (J-R) @ Q
    return A, G, J, R, Q


def get_u(t):
    u = torch.tensor([0, 0])
    return u


def get_R(x, perm):
    R = torch.zeros((x.shape[0], x.shape[0]))
    for i in range(int(x.shape[0] / 2)):
        if perm[i] == 0:
            R[2 * i + 1, 2 * i + 1] = 1
        if perm[i] == 1:
            R[2 * i + 1, 2 * i + 1] = 0.25 * x[2*i + 1]
        if perm[i] == 2:
            R[2 * i + 1, 2 * i + 1] = 0.25 * math.pow(x[2*i + 1], 2)

        if R[2 * i + 1, 2 * i + 1] > 1:
            print(R[2 * i + 1, 2 * i + 1])
    return R


def state_factory(J, Q, G, perm):
    J = torch.tensor(J)
    Q = torch.tensor(Q)
    G = torch.tensor(G)

    def state_trajectory(t, x):
        u = get_u(t)
        R = get_R(x, perm)
        return (J-R) @ Q @ x #+ G @ u

    return state_trajectory


def get_trajectory(state_trajectory, Q, t_span=[0, 6], timescale=20):
    t_eval = torch.linspace(t_span[0], t_span[1], int(timescale * (t_span[1] - t_span[0])))

    # get initial value
    x0 = np.random.rand(6) * 2 - 1
    radius = np.random.rand() * 0.9 + 0.1
    x0 = x0 / np.sqrt((x0 ** 2).sum()) * radius

    H0 = np.random.rand() * 0.45 + 0.05
    bot = x0.T @ Q @ x0
    x0 = np.sqrt((2*H0) / bot) * x0

    x0 = torch.tensor(x0.T)

    spring_ivp = odeint(func=state_trajectory, t=t_eval, y0=x0, rtol=1e-10, method='implicit_midpoint')
    dx = []
    for i in range(timescale * t_span[1]):
        dx.append(state_trajectory(t_eval[i], spring_ivp[i]))
    dx = torch.stack(dx)

    return spring_ivp, t_eval, dx


def get_dataset(dimension, seed, perm, samples=100, t_span=[0, 6], timescale=20, noise=0.):
    data = {'meta': locals()}
    np.random.seed(seed)

    A, G, J, R, Q = create_matrices(dimension)
    state_trajectory = state_factory(J=J, Q=Q, G=G, perm=perm)
    data['A'] = A
    data['G'] = G
    data['J'] = J
    data['R'] = R
    data['H'] = Q

    # train data
    xs, dxs, ts = [], [], []
    for s in range(samples):
        x, t, dx = get_trajectory(state_trajectory=state_trajectory, Q=Q, t_span=t_span, timescale=timescale)
        xs.append(x)
        dxs.append(dx)
        if s == 0:
            ts = t

    data['x'] = np.concatenate(xs).squeeze()
    data['dx'] = np.concatenate(dxs).squeeze()
    data['t'] = ts

    # test data
    xs, dxs = [], []
    for s in range(25):
        x, t, dx = get_trajectory(state_trajectory=state_trajectory, Q=Q, t_span=t_span, timescale=timescale)
        xs.append(x)
        dxs.append(dx)
        if s == 0:
            ts = t

    data['test_x'] = np.concatenate(xs).squeeze()
    data['test_dx'] = np.concatenate(dxs).squeeze()
    data['test_t'] = ts

    if noise != 0:
        data['x'] += data['x'] * np.random.randn(*data['x'].shape) * noise

    return data

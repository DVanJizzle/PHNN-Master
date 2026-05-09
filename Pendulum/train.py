import copy
import math

import torch, argparse
import numpy as np
import os

from NN import HMLP, RMLP
from HNN import HNN
from data import get_dataset, get_u
from torchdiffeq import odeint
import random


# For setting all hyperparameters
def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim_H', default=200, type=int, help='hidden dimension width of mlp for Gradient of H')
    parser.add_argument('--hidden_dim_R', default=200, type=int, help='hidden dimension width of mlp for R')
    parser.add_argument('--hidden_dim_G', default=125, type=int, help='hidden dimension width of mlp for G')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--patience', default=20, type=int, help='early stopping patience')
    parser.add_argument('--integration_steps', default=2, type=int, help='number of steps each ODE solve action uses')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--noise', default=0., type=float, help='noise for the training/ test data')
    parser.add_argument('--seed', default=0, type=int, help='randomness seed')
    parser.add_argument('--batch_size', default=64, type=int, help='size of batches for training')
    parser.add_argument('--weight_decay', default=0., type=float, help='weight decay regularization constant')
    parser.add_argument('--save_dir', default=os.path.dirname(os.path.abspath(__file__)), type=str,
                        help='where to save the trained model')
    parser.set_defaults(feature=True)
    return parser.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_printoptions(precision=8, sci_mode=True)

    nn_model = HMLP(args.input_dim, args.hidden_dim_H, output_dim=1, nonlinearity=args.nonlinearity)
    model_hamiltonian = HNN(args.input_dim, differentiable_model=nn_model)
    if args.weight_decay != 0:
        optim_hamiltonian = torch.optim.AdamW(model_hamiltonian.parameters(), args.learn_rate, weight_decay=args.weight_decay)
    else:
        optim_hamiltonian = torch.optim.Adam(model_hamiltonian.parameters(), args.learn_rate)
    #scheduler_Hamiltonian = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_hamiltonian, T_max=args.total_steps, eta_min=9e-4)

    model_R = RMLP(args.input_dim, args.hidden_dim_R, output_dim=3, nonlinearity=args.nonlinearity)
    if args.weight_decay != 0:
        optim_R = torch.optim.AdamW(model_R.parameters(), args.learn_rate, weight_decay=args.weight_decay)
    else:
        optim_R = torch.optim.Adam(model_R.parameters(), args.learn_rate)
    #scheduler_R = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_R, T_max=args.total_steps, eta_min=9e-4)

    data = get_dataset(seed=args.seed, noise=args.noise)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    x_test = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.tensor(data['dx'])
    dxdt_test = torch.tensor(data['test_dx'])
    t = data['t']
    t_test = data['test_t']

    # remove final values from inputs and initial values for computation of the loss
    x_input = trim_data(data['x'], t, end=True)
    x_input_test = trim_data(data['test_x'], t_test, end=True)
    x_loss = trim_data(data['x'], t, end=False)
    x_loss_test = trim_data(data['test_x'], t_test, end=False)

    x_input = torch.tensor(x_input, requires_grad=True, dtype=torch.float32)
    x_loss = torch.tensor(x_loss)
    x_input_test = torch.tensor(x_input_test, requires_grad=True, dtype=torch.float32)
    x_loss_test = torch.tensor(x_loss_test)

    J = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    G = torch.tensor([[0.], [1.]])
    chi = fun_factory(J=J, G=G, model_hamiltonian=model_hamiltonian, model_R=model_R)
    es_counter = 0
    best_test_loss = math.inf
    best_test_std = math.inf
    best_model_H = copy.deepcopy(model_hamiltonian)
    best_model_R = copy.deepcopy(model_R)
    batch_numbers = list(range(len(x_input)))
    batch_loss = torch.zeros(args.batch_size, 2)
    end_training = False
    step = 0
    test_loss = []
    train_loss = []
    lrs = []
    #lrs.append(scheduler_Hamiltonian.get_last_lr())

    # Training loop
    while True:
        random.shuffle(batch_numbers)

        for index, val in enumerate(batch_numbers):
            x_output_nn = odeint(func=chi, y0=x_input[val], t=torch.linspace(t[val % (len(t) - 1)], t[val % (len(t) - 1) + 1],
                                                                             args.integration_steps), rtol=1e-10, method='implicit_midpoint')[-1]
            batch_loss[index % args.batch_size] = (x_output_nn - x_loss[val]).square()

            if index > 0 and index % args.batch_size == args.batch_size-1:
                step += 1
                loss = batch_loss.mean()
                train_loss.append(loss.detach().numpy())
                loss.backward(retain_graph=True)
                optim_hamiltonian.step(); optim_R.step()
                optim_hamiltonian.zero_grad(); optim_R.zero_grad()
                batch_loss = torch.zeros(args.batch_size, 2)
                #lrs.append(scheduler_Hamiltonian.get_last_lr())

                # Early stopping
                loss, std = get_MSE_state_loss(x_input_test, x_loss_test, t_test, chi, args)
                #scheduler_Hamiltonian.step()
                #scheduler_R.step()
                test_loss.append(loss.detach().numpy())
                if step % 5 == 0:
                    print('Epoch: ', step, ' train loss: ', train_loss[-1])
                    print('Epoch: ', step, ' test loss: ', loss)
                if loss < best_test_loss:
                    best_test_loss = loss
                    best_test_std = std / np.sqrt(x_test.shape[0])
                    es_counter = 0
                    best_model_H = copy.deepcopy(model_hamiltonian)
                    best_model_R = copy.deepcopy(model_R)
                else:
                    es_counter += 1
                    if es_counter >= args.patience:
                        chi = fun_factory(J, G, best_model_H, best_model_R)
                        print('Stopped Early: Best Hamiltonian found at epoch ', step - args.patience)
                        end_training = True
                        break

                if step == args.total_steps:
                    chi = fun_factory(J, G, best_model_H, best_model_R)
                    break

        # Early stopping patience limit or maximum epoch reached
        if step == args.total_steps:
            print("Training ended due to reaching maximum epoch amount")
            end_training = True

        if end_training:

            loss, std = get_MSE_state_loss(x_input, x_loss, t, chi, args)
            print("Final train loss: ", loss, " +/-",  std / np.sqrt(x.shape[0]))
            print("Final test loss: ", best_test_loss, " +/-",  best_test_std)
            data['test_loss'] = test_loss
            data['train_loss'] = train_loss
            break

    #end of training
    chi = fun_factory(J, G, best_model_H, best_model_R)
    data['lr'] = lrs

    train_rhs_loss, train_rhs_std = get_MSE_rhs_loss(x, dxdt, chi, t)
    test_rhs_loss, test_rhs_std = get_MSE_rhs_loss(x_test, dxdt_test, chi, t_test)

    print("Train rhs loss: ", train_rhs_loss, "+/-", train_rhs_std / np.sqrt(x.shape[0]))
    print("Test rhs loss: ", test_rhs_loss, "+/-", test_rhs_std / np.sqrt(x_test.shape[0]))
    return best_model_H, best_model_R, chi, data, args


def fun_factory(J, G, model_hamiltonian, model_R):
    def chi(t, x):
        u = get_u(t).unsqueeze(0)
        L_result = model_R(x)
        L_upper_triangle = torch.zeros(2, 2)
        L_upper_triangle[0, 0] = L_result[0]
        L_upper_triangle[0, 1] = L_result[1]
        L_upper_triangle[1, 1] = L_result[2]
        R = L_upper_triangle @ L_upper_triangle.T
        x_dot = (J - R) @ model_hamiltonian.time_derivative(x) + G @ u
        return x_dot

    return chi


def trim_data(x, t, end):
    deletion_indexes = []
    if end:
        for i in range(len(t) - 1, len(x), len(t)):
            deletion_indexes.append(i)
    else:
        for i in range(0, len(x), len(t)):
            deletion_indexes.append(i)
    for i in sorted(deletion_indexes, reverse=True):
        x = np.delete(x, i, axis=0)
    return x


def get_MSE_rhs_loss(x, dxdt, chi, t):
    dx_hat = torch.zeros(x.shape)
    j = 0
    for i, val in enumerate(x):
        dx_hat[i] = chi(t[j], val)
        j += 1
        if j == len(t):
            j = 0

    return (dxdt - dx_hat).square().mean(), (dxdt - dx_hat).square().std()


def get_MSE_state_loss(x, x_loss, t, chi, args):
    x_output = torch.zeros(x.shape)
    j = 0
    for i, val in enumerate(x):
        x_output[i] = \
        odeint(func=chi, y0=val, t=torch.linspace(t[j], t[j + 1], args.integration_steps), rtol=1e-10,
               method='implicit_midpoint')[-1]
        j += 1
        if j == len(t) - 1:
            j = 0

    return (x_output - x_loss).square().mean(), (x_output - x_loss).square().std()


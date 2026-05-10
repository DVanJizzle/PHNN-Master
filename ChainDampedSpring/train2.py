import copy
import math
import numpy as np
import torch
from NN import HMLP, RMLP
from HNN import HNN
from data import get_dataset, get_u
from train import fun_factory, get_MSE_rhs_loss
import random


def train(args):
    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_printoptions(precision=8, sci_mode=True)

    nn_model = HMLP(2 * args.input_dim, args.hidden_dim_H, output_dim=1, nonlinearity=args.nonlinearity)
    model_hamiltonian = HNN(2 * args.input_dim, differentiable_model=nn_model)
    if args.weight_decay != 0:
        optim_hamiltonian = torch.optim.AdamW(model_hamiltonian.parameters(), args.learn_rate,
                                              weight_decay=args.weight_decay)
    else:
        optim_hamiltonian = torch.optim.Adam(model_hamiltonian.parameters(), args.learn_rate)
    #scheduler_Hamiltonian = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim_Hamiltonian, patience=10, factor=0.5, threshold=1e-7)

    output_dim_R = int((2 * args.input_dim * (2 * args.input_dim + 1)) / 2)
    model_R = RMLP(2 * args.input_dim, args.hidden_dim_R, output_dim=output_dim_R, nonlinearity=args.nonlinearity)
    if args.weight_decay != 0:
        optim_R = torch.optim.AdamW(model_R.parameters(), args.learn_rate, weight_decay=args.weight_decay)
    else:
        optim_R = torch.optim.Adam(model_R.parameters(), args.learn_rate)
    #scheduler_R = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim_R, T_max=args.schedulerepochs, eta_min=1e-4)

    data = get_dataset(dimension=args.input_dim, seed=args.seed, noise=args.noise, perm=args.perm)
    x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float32)
    x_test = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.tensor(data['dx'])
    dxdt_test = torch.tensor(data['test_dx'])
    t = data['t']
    t_test = data['test_t']

    J = torch.tensor(data['J']).float()
    G = torch.tensor(data['G']).float()
    chi = fun_factory(J=J, G=G, model_hamiltonian=model_hamiltonian, model_R=model_R, input_dim=args.input_dim)
    es_counter = 0
    best_test_loss = math.inf
    best_test_std = math.inf
    best_model_H = copy.deepcopy(model_hamiltonian)
    best_model_R = copy.deepcopy(model_R)
    batch_numbers = list(range(len(x)))
    batch_loss = torch.zeros(args.batch_size, args.input_dim * 2)
    end_training = False
    step = 0
    test_loss = []
    train_loss = []
    lrs = []
    # lrs.append(scheduler_Hamiltonian.get_last_lr())

    # Training loop
    while True:
        random.shuffle(batch_numbers)

        for index, val in enumerate(batch_numbers):
            x_output_nn = chi(t[val % (len(t) - 1)], x[val])
            batch_loss[index % args.batch_size] = (x_output_nn - dxdt[val]).square()

            if index > 0 and index % args.batch_size == args.batch_size - 1:
                step += 1
                loss = batch_loss.mean()
                train_loss.append(loss.detach().numpy())
                loss.backward(retain_graph=True)
                optim_hamiltonian.step(); optim_hamiltonian.zero_grad()
                optim_R.step(); optim_R.zero_grad()
                batch_loss = torch.zeros(args.batch_size, args.input_dim * 2)
                #lrs.append(scheduler_Hamiltonian.get_last_lr())

                # Early stopping
                loss, std = get_MSE_rhs_loss(x_test, dxdt_test, chi, t_test)
                test_loss.append(loss.detach().numpy())
                #scheduler_Hamiltonian.step()
                #scheduler_R.step()
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
                        chi = fun_factory(J, G, best_model_H, best_model_R, args.input_dim)
                        print('Stopped Early: Best Hamiltonian found at epoch ', step - args.patience)
                        end_training = True
                        break

                if step == args.total_steps:
                    chi = fun_factory(J, G, best_model_H, best_model_R, args.input_dim)
                    break

        # Early stopping patience limit or maximum epoch reached
        if step == args.total_steps:
            print("Training ended due to reaching maximum epoch amount")
            end_training = True

        if end_training:
            loss, std = get_MSE_rhs_loss(x, dxdt, chi, t)
            print("Final train loss: ", loss, " +/-", std / np.sqrt(x.shape[0]))
            print("Final test loss: ", best_test_loss, " +/-", best_test_std)
            data['test_loss'] = test_loss
            data['train_loss'] = train_loss
            break

    # end of training
    chi = fun_factory(J, G, best_model_H, best_model_R, args.input_dim)
    data['lr'] = lrs

    return best_model_H, best_model_R, chi, data, args

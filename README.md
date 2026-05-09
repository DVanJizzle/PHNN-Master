# Port-Hamiltonian Neural Networks

This repository contains the code used for writing the master's thesis __Port-Hamiltonian Neural Networks: Data-Driven Modeling of Energy-Conserving Systems__.

- Each folder contains a module data.py for simulating trajectories and two train modules.
- The module train.py uses a state loss, while train2.py uses a RHS loss. Other than that, they function identically.
- The NN hyperparameters can be set at the top of each train module.
- To start training, execute 
    - args = get(args)
    - modelH, model_R, chi, data, args = train(args)
- The network width can be determined in NN.py
- HNN.py contains the structure for the Hamiltonian neural network (This and other parts of the code are directly based on: https://github.com/greydanus/hamiltonian-nn#)
- In addition to this code, the module torchdiffeq is needed, which can be downloaded here: https://github.com/rtqichen/torchdiffeq


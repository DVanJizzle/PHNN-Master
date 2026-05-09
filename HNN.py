import torch

class HNN(torch.nn.Module): # this module needs some cleaning
    def __init__(self, input_dim, differentiable_model,
                    assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords

    def forward(self, x):
        y = self.differentiable_model(x)
        return y

    def time_derivative(self, x):
        H = self.forward(x)  # traditional forward pass
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        return dH
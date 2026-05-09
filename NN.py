import torch


class HMLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(HMLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear20 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear21 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear22 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear23 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear24 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear25 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear26 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

    for l in [self.linear1, self.linear20, self.linear21, self.linear22, self.linear23, self.linear3]:
      torch.nn.init.xavier_uniform_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity(self.linear1(x))
    h = self.nonlinearity(self.linear20(h))
    h = self.nonlinearity(self.linear21(h))
    h = self.nonlinearity( self.linear22(h) )
    h = self.nonlinearity( self.linear23(h) )
    #h = self.nonlinearity( self.linear24(h) )
    #h = self.nonlinearity( self.linear25(h) )
    #h = self.nonlinearity( self.linear26(h) )
    return self.linear3(h)

class RMLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(RMLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear20 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear21 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear22 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear23 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear24 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear25 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear26 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

    for l in [self.linear1, self.linear20, self.linear21, self.linear22, self.linear23, self.linear3]:
      torch.nn.init.xavier_uniform_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity(self.linear1(x))
    h = self.nonlinearity(self.linear20(h))
    h = self.nonlinearity(self.linear21(h))
    h = self.nonlinearity( self.linear22(h) )
    h = self.nonlinearity( self.linear23(h) )
    #h = self.nonlinearity( self.linear24(h) )
    #h = self.nonlinearity( self.linear25(h) )
    #h = self.nonlinearity( self.linear26(h) )
    return self.linear3(h)

class GMLP(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
    super(GMLP, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
    self.linear20 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear21 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear22 = torch.nn.Linear(hidden_dim, hidden_dim)
    #self.linear23 = torch.nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

    for l in [self.linear1, self.linear20, self.linear21, self.linear3]:
      torch.nn.init.xavier_uniform_(l.weight)

    self.nonlinearity = choose_nonlinearity(nonlinearity)

  def forward(self, x):
    h = self.nonlinearity(self.linear1(x))
    h = self.nonlinearity(self.linear20(h))
    h = self.nonlinearity(self.linear21(h))
    #h = self.nonlinearity( self.linear22(h) )
    #h = self.nonlinearity( self.linear23(h) )
    return self.linear3(h)

def choose_nonlinearity(name):
  nl = None
  if name == 'tanh':
    nl = torch.tanh
  elif name == 'relu':
    nl = torch.relu
  elif name == 'sigmoid':
    nl = torch.sigmoid
  elif name == 'softplus':
    nl = torch.nn.functional.softplus
  elif name == 'selu':
    nl = torch.nn.functional.selu
  elif name == 'elu':
    nl = torch.nn.functional.elu
  elif name == 'swish':
    nl = lambda x: x * torch.sigmoid(x)
  else:
    raise ValueError("nonlinearity not recognized")
  return nl

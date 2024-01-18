import torch
import torch.nn as nn
from kstv import *

# add_bus = add_bus =  [100, 49, 77, 89, 12, 59, 37, 32, 19, 104, 97, 67, 112, 41, 3, 94, 6, 28, 79, 13]
X, Y = data( output='v_all')
device =  torch.device('cuda')

X_training, X_val, X_test, y_training, y_val, y_test = split_normalise(X, Y, Pinj, Qinj)

x_training = torch.from_numpy( np.nan_to_num(X_training).astype(np.float32) ).to(device)
x_val = torch.from_numpy( np.nan_to_num(X_val).astype(np.float32) ).to(device)
x_test = torch.from_numpy( np.nan_to_num(X_test).astype(np.float32) ).to(device)

y_training = torch.from_numpy( np.nan_to_num(y_training).astype(np.float32) ).to(device)
y_val = torch.from_numpy( np.nan_to_num(y_val).astype(np.float32) ).to(device)
y_test = torch.from_numpy( np.nan_to_num(y_test).astype(np.float32) ).to(device)

net = FullyConnectedNetwork(x_training.shape[1], y_training.shape[1]).to(device)
net.load_state_dict(torch.load("../checkpoints/breezy-teal-fousek.pt"))
net.eval()

loss=torch.nn.MSELoss().to(device)

out= net(x_test)
error= loss(out,y_test)
print(f"test_error {error}")

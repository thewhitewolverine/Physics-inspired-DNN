from import_data import *
from noise_models import *
from PICDNN_extended import *
from Branch_flow_xy_split import *
# from generate_xy import *
import numpy as np
import matplotlib.pyplot as plt
# from volt_to_p_est import *
# from VI_to_P import *
# from linear_models import *
from feature_bins import std_bins_bflow
from IPython.display import clear_output

from scaler import *
from extended_loss import *
from final_layer_weights import final_layer_weights

VR, VI, bflow, Pinj, Qinj, CFR, CFI, CTR, CTI = import_curr_branch_data()
def data(addbus=[], output = "vi_all"):
    # Getting X and Y 
    
    x_vr, x_vi, x_cr, x_ci,  ytot = Branch_flow_custom_xy_split(bflow, VR, VI, CFR, CFI, CTR, CTI, Pinj, Qinj, output = output,  num_bus = 11, add_bus = addbus)
    # Adding noise 
    x_vr_n, x_vi_n = GMM_real_imag(x_vr, x_vi)
    x_cr_n, x_ci_n = GMM_real_imag(x_cr, x_ci)
    # bflow_n = noise_gaussian(X, range1, mu1)
    # bflow_n = NoisyMag(x_bflow.values)
 

    # X_p = VI_to_P(x_vm_n, x_va_n, x_cm_n, x_ca_n, num_bus = 11, add_bus = [])
    X_v = np.concatenate((x_vr_n, x_vi_n, x_cr_n, x_ci_n), axis=1)
    Y_p = ytot
    return X_v, Y_p

def mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, multioutput
    )
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)

def test_error(self, x_test, y_test):

    yhat = self.model.predict(x_test)
    # l, b = yhat.shape

    # no_load_buses = [5, 9, 30, 37, 38, 63, 64, 68, 71, 81]
    # no_load_index = [i-1 for i in no_load_buses]

    if self.y_normalise:
        yhat_a = self.y_scaler.inverse_transform(yhat)
        ytest_a = self.y_scaler.inverse_transform(y_test)
    else:
        yhat_a = yhat
        ytest_a = y_test

    if self.save:
        np.savetxt('yhat.csv', yhat_a, delimiter = ',')
        # np.savetxt('ytest.csv', ytest_a, delimiter = ',')

    
    mse = [0 for i in range(ytest_a.shape[1])]
    for i in range(ytest_a.shape[1]):
        mse[i] = mean_squared_error(yhat_a[:,i], ytest_a[:,i])

    mape = [0 for i in range(ytest_a.shape[1])]
    for i in range(ytest_a.shape[1]):
        mape[i] = mean_absolute_percentage_error(yhat_a[:,i], ytest_a[:,i])

    r2 = [0 for i in range(ytest_a.shape[1])]
    for i in range(ytest_a.shape[1]):
        r2[i] = r2_score(ytest_a[:,i], yhat_a[:,i])


    lengths = [118, 118, 186,186,186,186]
    # mse2 = np.matmul(mse, self.last_layer_weights)
    return mse, mape,  r2

def split_normalise(X, y, Pinj, Qinj):
    x_scaler = scaler(X.shape[1])
    y_scaler = scaler(y.shape[1])
    x_train, x_test, y_train, y_test, _, Ptest, _, Qtest = train_test_split(X, y, Pinj, Qinj, test_size=0.2)
    x_training, x_val, y_training, y_val = train_test_split(x_train, y_train, test_size=0.1)

    # if self.save:
    #     np.savetxt('ptest.csv', Ptest, delimiter = ',')
    #     np.savetxt('qtest.csv', Qtest, delimiter = ',')


    x_train_scaled = x_scaler.fit_transform(x_training)
    x_val_scaled = x_scaler.transform(x_val)
    x_test_scaled = x_scaler.transform(x_test)


    # if self.y_normalise:
    #     y_train_scaled = self.y_scaler.fit_transform(y_training)
    #     y_val_scaled = self.y_scaler.transform(y_val)
    #     y_test_scaled = self.y_scaler.transform(y_test)
    # else:
    y_train_scaled = y_training
    y_val_scaled = y_val
    y_test_scaled = y_test

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled


def power_injection_loss(self, x, xhat):
    last_layer_weights = torch.from_numpy( final_layer_weights().astype(np.float32) )
    y = torch.matmul(x, last_layer_weights)
    yhat = torch.matmul(xhat, last_layer_weights)
    e = torch.square( (y-yhat) )
    b_inv1 = torch.inverse(torch.matmul(last_layer_weights, tf.transpose(last_layer_weights)))
    b_inv = tf.matmul(tf.transpose(last_layer_weights), b_inv1)

    e_back = tf.matmul(e, b_inv)
    # print(e_back.shape)
    return e_back

def extended_loss(self, x, xhat):
    # regular MSE
    squared_difference = tf.square(x - xhat)

    # calculate power flow and get pinj loss
    PFF, PFT, PFF_actual, PFT_actual = vi_to_power_flow(x, xhat)
    bflow_hat = reorder(PFF, PFT)
    bflow_actual = reorder(PFF_actual, PFT_actual)

    # get overall power loss
    pinj_loss = self.power_injection_loss(bflow_actual, bflow_hat)

    pf_loss = state_loss_power_flow(x, xhat, pinj_loss)/100
    pf_loss = tf.cast(pf_loss, tf.float32)

    extended_loss = tf.add(squared_difference, pf_loss)
    return extended_loss


# MODEL
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# class MyDataset(Dataset):
#     def __init__(self):
#         pmu_loc = {
#             11: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81] ,
#             13: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] ,
#             32: [1, 5, 9, 12,13,17,21,23,26,28,34,37,41,45,49,53,56,62,63,68,71,75,77,80,85,86,90,94,101,105,110,114],
#             118: np.arange(1,119)
#         }

#         add_bus = [x for x in pmu_loc[32] if x not in pmu_loc[11]]

#         self.X_v, self.Y_p = data(add_bus, 'vi_all')    

#     def __len__(self):
#         return len(self.X_v)

#     def __getitem__(self,idx):
#         return X_v



class FullyConnectedNetwork(nn.Module):
    def __init__(self, in_size):
        super(FullyConnectedNetwork,self).__init__()
        self.hidden1=nn.Sequential(
                nn.Linear(in_features=in_size,out_features=150,bias=True),
                nn.ReLU())
        self.hidden2=nn.Sequential(
                nn.Linear(in_features=150,out_features=100,bias=True),
                nn.ReLU())
        self.hidden3=nn.Sequential(
                nn.Linear(in_features=100,out_features=300,bias=True),
                nn.ReLU())
        self.hidden4=nn.Sequential(
                nn.Linear(in_features=300,out_features=500,bias=True),
                nn.ReLU())

        self.hidden5=nn.Sequential(
                nn.Linear(in_features=500,out_features=750,bias=True),
                nn.ReLU())

        # self.hidden3=nn.Sequential(
        #         nn.Linear(in_features=100,out_features=50,bias=True),
        #         nn.ReLU())

        self.predict= nn.Linear(in_features=750,out_features=980,bias=True)

    def forward(self,x):
        x=self.hidden1(x)
        x=self.hidden2(x)
        x=self.hidden3(x)
        x=self.hidden4(x)
        x=self.hidden5(x)
        x=self.predict(x)
        return x


import namegenerator
if __name__ == '__main__':
    pmu_loc = {
        11: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81] ,
        13: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] ,
        32: [1, 5, 9, 12,13,17,21,23,26,28,34,37,41,45,49,53,56,62,63,68,71,75,77,80,85,86,90,94,101,105,110,114],
        118: np.arange(1,119)
    }

    #add_bus = [x for x in pmu_loc[32] if x not in pmu_loc[11]]

    add_bus = add_bus =  [100, 49, 77, 89, 12, 59, 37, 32, 19, 104, 97, 67, 112, 41, 3, 94, 6, 28, 79, 13]

    device =  torch.device('cuda')

    X, Y = data(add_bus, 'vi_all')        
    X_training, X_val, X_test, y_training, y_val, y_test = split_normalise(X, Y, Pinj, Qinj)

    x_training = torch.from_numpy( np.nan_to_num(X_training).astype(np.float32) )
    x_val = torch.from_numpy( np.nan_to_num(X_val).astype(np.float32) ).to(device)
    x_test = torch.from_numpy( np.nan_to_num(X_test).astype(np.float32) ).to(device)

    y_training = torch.from_numpy( np.nan_to_num(y_training).astype(np.float32) )
    y_val = torch.from_numpy( np.nan_to_num(y_val).astype(np.float32) ).to(device)
    y_test = torch.from_numpy( np.nan_to_num(y_test).astype(np.float32) ).to(device)

    batch_size= 1500

    # print(x_training.shape, y_training.shape)
    # print(x_val.shape, y_val.shape)
    # print(x_test.shape, y_test.shape)

    train_dataset=TensorDataset(x_training,y_training)
    train_iter=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


    # print(len(dataset), len(train_iter))    
    net = FullyConnectedNetwork(x_training.shape[1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr = 1e-3)
    loss=torch.nn.MSELoss().to(device)    

    best_valloss= 0.0015

    model_name= namegenerator.gen()
    record_file= f"../checkpoints/{model_name}_log.txt"
    out_file= open(record_file,'w')
    
    out_file.write( str(net) )

    for epoch in range(200):
        for index, data in enumerate(train_iter):
            x,y= data
            x,y= x.to(device), y.to(device)
            out= net(x)
            loss_ = loss(out,y)

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
        
        with torch.no_grad():            
            out= net(x_val)
            vloss= loss(out,y_val)
            
            if vloss< best_valloss:
                best_valloss= vloss
                torch.save(net.state_dict(), f"../checkpoints/{model_name}.pt")
            out_file.write((f"{epoch}: \t {vloss}\n"))
            print((f"{epoch}: \t {vloss}"))
    
    
    test_out= net(x_test)
    error= loss(test_out,y_test)
    
    out_file.write(f"\n\n TEST ERROR: {error}")
    print(f"\n\n TEST ERROR: {error}")

    out_file.close()



    # loss=torch.nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(net.parameters(),lr = 0.1)

    # dnn = Deep_Network(layers=5, nodes=int(X_v.shape[1]*1.25), lr = 1e-4, epoch = 200,
    #                 batch_norm = True, dropout = 0)
    # # dnn.y_normalise = True
    # dnn.loss = "mse"
    # mse, mape, r2 = dnn.model_parse(X_v,Y_p,Pinj, Qinj, ntest=1)


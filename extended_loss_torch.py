import torch
import numpy as np

device =  torch.device('cuda')

def create_A_array(bdata, n_buses):
    n = bdata.shape[0]
    A = np.zeros((n_buses, n))
    for i in range(n_buses):
        for j in range(n):
            if bdata[j, 0] == i+1:
                A[i, j] = 1
    return A


def create_C_array(bdata, n_buses):
    n = bdata.shape[0]
    C = np.zeros((n_buses, n))
    for i in range(n_buses):
        for j in range(n):
            if bdata[j, 1] == i+1:
                C[i, j] = 1
    return C

def create_B_array(bdata):
    n = bdata.shape[0]
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = 1
    return B


def power_from_vi(vars, A, B, C):
    Pff_v1  = torch.matmul(vars[0], A)
    Pff_v2  = torch.matmul(vars[2], B)
    Pff_v3  = torch.matmul(vars[1], A)
    Pff_v4  = torch.matmul(vars[3], B)

    Pft_v1  = torch.matmul(vars[0], C)
    Pft_v2  = torch.matmul(vars[4], B)
    Pft_v3  = torch.matmul(vars[1], C)
    Pft_v4  = torch.matmul(vars[5], B)

    # no need to multiply from SBase, since we are calculating loss
    PFF = torch.multiply(torch.add(torch.multiply(Pff_v1, Pff_v2), torch.multiply(Pff_v3, Pff_v4)), 1)
    PFT = torch.multiply(torch.add(torch.multiply(Pft_v1, Pft_v2), torch.multiply(Pft_v3, Pft_v4)), 1)

    return PFF, PFT


bdata = np.genfromtxt("System_data/bdata.csv", delimiter=',')
A = torch.from_numpy(create_A_array(bdata,118).astype(np.float32) ).to(device)
B = torch.from_numpy(create_B_array(bdata).astype(np.float32) ).to(device)
C = torch.from_numpy(create_C_array(bdata,118).astype(np.float32) ).to(device)

# VR, VI, CFR, CFI, CTR, CTI
lengths = [118, 118, 186,186,186,186]

def vi_to_power_flow(x, xhat):    
    sample_input = xhat
    vars = torch.split(sample_input, lengths, dim=1)
    vars_actual = torch.split(x, lengths, dim=1)

    PFF_actual ,PFT_actual = power_from_vi(vars_actual, A, B, C)
    PFF, PFT = power_from_vi(vars, A, B, C)

    return PFF, PFT, PFF_actual, PFT_actual    



def state_loss_power_flow(x, xhat, p_inj_loss):    
    sample_input = xhat 
    vars_np = torch.split(sample_input, lengths, dim=1)
    # vars_actual = tf.split(x, lengths, axis=1)
    PFF, PFT, PFF_actual, PFT_actual = vi_to_power_flow(x, xhat)

    ePF1 = PFF - PFF_actual
    ePT1 = PFT - PFT_actual

    ePFFPinj, ePFTPinj = reorder_inverse(p_inj_loss)
    ePF2 = ePFFPinj
    ePT2 = ePFTPinj

    ePF = ePF1 + ePF2
    ePT = ePT1 + ePT2

    # print([sample_input.shape[0], lengths[0]])

    pfl_vr_1 = torch.zeros([sample_input.shape[0], lengths[0]]).to(device)
    pfl_vr_2 = torch.zeros([sample_input.shape[0], lengths[0]]).to(device)
    pfl_vi_1 = torch.zeros([sample_input.shape[0], lengths[1]]).to(device)
    pfl_vi_2 = torch.zeros([sample_input.shape[0], lengths[1]]).to(device)
    pfl_cfr = torch.zeros([sample_input.shape[0], lengths[2]]).to(device)
    pfl_cfi = torch.zeros([sample_input.shape[0], lengths[3]]).to(device)
    pfl_ctr = torch.zeros([sample_input.shape[0], lengths[4]]).to(device)
    pfl_cti = torch.zeros([sample_input.shape[0], lengths[5]]).to(device)


    # vars_np = []
    # for i in vars:
    #     vars_np.append(i.numpy())

    for i in range(bdata.shape[0]):
        # current index is i, so is power 
        fbus = int(bdata[i,0])-1
        tbus = int(bdata[i,1])-1

        # loss of fbus updated with i-th from current
        pfl_vr_1[:,fbus] = pfl_vr_1[:,fbus] + ePF[:,i]*vars_np[2][:,i]/(1+ vars_np[2][:,i]**2)
        pfl_vi_1[:,fbus] = pfl_vi_1[:,fbus] + ePF[:,i]*vars_np[3][:,i]/(1+ vars_np[3][:,i]**2)
        
        # loss of tbus updated with ith to current
        pfl_vr_2[:,tbus] = pfl_vr_2[:,tbus] + ePT[:,i]*vars_np[4][:,i]/(1+vars_np[4][:,i]**2)
        pfl_vi_2[:,tbus] = pfl_vi_2[:,tbus] + ePT[:,i]*vars_np[5][:,i]/(1+vars_np[5][:,i]**2)

        # loss of ith current updated with fbus voltage and tbus voltage
        pfl_cfr[:,i] = pfl_cfr[:,i] + ePF[:,i]*vars_np[0][:,fbus]/(1+vars_np[0][:,fbus]**2)

        pfl_cfi[:,i] = pfl_cfi[:,i] + ePF[:,i]*vars_np[1][:,fbus]/(1+vars_np[1][:,fbus]**2)

        pfl_ctr[:,i] = pfl_ctr[:,i] + ePF[:,i]*vars_np[0][:,tbus]/(1+vars_np[0][:,tbus]**2)

        pfl_cti[:,i] = pfl_cti[:,i] + ePF[:,i]*vars_np[1][:,tbus]/(1+vars_np[1][:,tbus]**2)


    vr_loss = pfl_vr_1 + pfl_vr_2
    vi_loss = pfl_vi_1 + pfl_vi_2

    # loss_for_sample_b = torch.concat([vr_loss, vi_loss, pfl_cfr, pfl_cfi, pfl_ctr, pfl_cti], dim=1)
    # return loss_for_sample_b

    V_loss= torch.concat([vr_loss, vi_loss],dim=1)
    P_loss= torch.concat([pfl_cfr, pfl_cfi, pfl_ctr, pfl_cti], dim=1)
    return V_loss/100,P_loss/100
    


def reorder(PFF, PFT):
    # Split PFF and PFT into individual columns
    PFF_columns = torch.split(PFF, PFF.shape[1], dim=1)
    PFT_columns = torch.split(PFT, PFT.shape[1], dim=1)

    # Alternate between PFF and PFT columns
    alternating_columns = [col for pair in zip(PFF_columns, PFT_columns) for col in pair]

    # Stack the alternating columns together
    bflow = torch.concat(alternating_columns, axis=1)
    return bflow  

def reorder_inverse(ebflow):
    # Get all the even columns (0-indexed, so even indices will be 0, 2, 4, ...)
    PFF = ebflow[:, ::2]

    # Get all the odd columns (0-indexed, so odd indices will be 1, 3, 5, ...)
    PFT = ebflow[:, 1::2]

    return PFF, PFT
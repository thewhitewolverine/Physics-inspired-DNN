import tensorflow as tf
import numpy as np


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
    Pff_v1  = tf.linalg.matmul(vars[0], A)
    Pff_v2  = tf.linalg.matmul(vars[2], B)
    Pff_v3  = tf.linalg.matmul(vars[1], A)
    Pff_v4  = tf.linalg.matmul(vars[3], B)

    Pft_v1  = tf.linalg.matmul(vars[0], C)
    Pft_v2  = tf.linalg.matmul(vars[4], B)
    Pft_v3  = tf.linalg.matmul(vars[1], C)
    Pft_v4  = tf.linalg.matmul(vars[5], B)


    PFF = tf.multiply(tf.add(tf.multiply(Pff_v1, Pff_v2), tf.multiply(Pff_v3, Pff_v4)), 1) # no need to multiply from SBase, since we are calculating loss
    PFT = tf.multiply(tf.add(tf.multiply(Pft_v1, Pft_v2), tf.multiply(Pft_v3, Pft_v4)), 1)

    return PFF, PFT

def vi_to_power_flow(x, xhat):
    bdata = np.genfromtxt(r"System_data\bdata.csv", delimiter=',')
    A = tf.convert_to_tensor(create_A_array(bdata,118), dtype=tf.float32)
    B = tf.convert_to_tensor(create_B_array(bdata), dtype=tf.float32)
    C = tf.convert_to_tensor(create_C_array(bdata, 118), dtype=tf.float32)

    
    lengths = [118, 118, 186,186,186,186] # VR, VI, CFR, CFI, CTR, CTI
    
    sample_input = xhat
    vars = tf.split(sample_input, lengths, axis=1)
    vars_actual = tf.split(x, lengths, axis=1)

    PFF_actual ,PFT_actual = power_from_vi(vars_actual, A, B, C)
    PFF, PFT = power_from_vi(vars, A, B, C)

    return PFF, PFT, PFF_actual, PFT_actual


def state_loss_power_flow(x, xhat, p_inj_loss):
    bdata = np.genfromtxt(r"System_data\bdata.csv", delimiter=',')
    lengths = [118, 118, 186,186,186,186] # VR, VI, CFR, CFI, CTR, CTI
    sample_input = xhat 
    vars = tf.split(sample_input, lengths, axis=1)
    # vars_actual = tf.split(x, lengths, axis=1)
    PFF, PFT, PFF_actual, PFT_actual = vi_to_power_flow(x, xhat)

    ePF1 = (tf.subtract(PFF, PFF_actual)).numpy()
    ePT1 = (tf.subtract(PFT, PFT_actual)).numpy()

    ePFFPinj, ePFTPinj = reorder_inverse(p_inj_loss)
    ePF2 = ePFFPinj.numpy()
    ePT2 = ePFTPinj.numpy()

    ePF = ePF1 + ePF2
    ePT = ePT1 + ePT2

    pfl_vr_1 = np.zeros([sample_input.shape[0], lengths[0]])
    pfl_vr_2 = np.zeros([sample_input.shape[0], lengths[0]])
    pfl_vi_1 = np.zeros([sample_input.shape[0], lengths[1]])
    pfl_vi_2 = np.zeros([sample_input.shape[0], lengths[1]])
    pfl_cfr = np.zeros([sample_input.shape[0], lengths[2]])
    pfl_cfi = np.zeros([sample_input.shape[0], lengths[3]])
    pfl_ctr = np.zeros([sample_input.shape[0], lengths[4]])
    pfl_cti = np.zeros([sample_input.shape[0], lengths[5]])

    vars_np = []
    for i in vars:
        vars_np.append(i.numpy())

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


    vr_loss = tf.convert_to_tensor(pfl_vr_1 + pfl_vr_2)
    vi_loss = tf.convert_to_tensor(pfl_vi_1 + pfl_vi_2)

    cfr_loss = tf.convert_to_tensor(pfl_cfr)
    cfi_loss = tf.convert_to_tensor(pfl_cfi)
    ctr_loss = tf.convert_to_tensor(pfl_ctr)
    cti_loss = tf.convert_to_tensor(pfl_cti)

    loss_for_sample_b = tf.concat([vr_loss, vi_loss, cfr_loss, cfi_loss, ctr_loss, cti_loss], axis=1)
    
    return loss_for_sample_b


def reorder(PFF, PFT):
    # Split PFF and PFT into individual columns
    PFF_columns = tf.split(PFF, PFF.shape[1], axis=1)
    PFT_columns = tf.split(PFT, PFT.shape[1], axis=1)

    # Alternate between PFF and PFT columns
    alternating_columns = [col for pair in zip(PFF_columns, PFT_columns) for col in pair]

    # Stack the alternating columns together
    bflow = tf.concat(alternating_columns, axis=1)
    return bflow  

def reorder_inverse(ebflow):
    # Get all the even columns (0-indexed, so even indices will be 0, 2, 4, ...)
    PFF = ebflow[:, ::2]

    # Get all the odd columns (0-indexed, so odd indices will be 1, 3, 5, ...)
    PFT = ebflow[:, 1::2]

    return PFF, PFT
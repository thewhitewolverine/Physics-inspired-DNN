import numpy as np
import torch 



def extened_loss_v_pinj():
    return


def current_from_voltage(vr_hat, vi_hat, bdata):
    cfr_vr_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))
    cfr_vi_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))

    cfi_vr_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))
    cfi_vi_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))

    ctr_vr_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))
    ctr_vi_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))

    cti_vr_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))
    cti_vi_matrix  = np.zeros((vr_hat.shape[0], bdata.shape[0]))

    for idx,branch in enumerate(bdata):
        fbus = int(branch[0])-1
        tbus = int(branch[1])-1
        R = branch[2]
        X = branch[3]
        B = branch[4]
        TAU = branch[5]

        AR = R/(TAU**2 *(R**2 + X**2))
        AI = B/(2*TAU**2) - X/(TAU**2 *(R**2 + X**2))

        BR = -R/(TAU*(R**2 + X**2))
        BI = X/(TAU*(R**2 + X**2))

        DR = R/(R**2 + X**2)
        DI = B/2 - X/(R**2 + X**2)


        cfr_vr_matrix[fbus,idx] = AR
        cfr_vr_matrix[tbus,idx] = BR

        cfr_vi_matrix[fbus,idx] = -AI
        cfr_vi_matrix[tbus,idx] = -BI

        cfi_vr_matrix[fbus,idx] = AI
        cfi_vr_matrix[tbus,idx] = BI

        cfi_vi_matrix[fbus,idx] = AR
        cfi_vi_matrix[tbus,idx] = BR

        ctr_vr_matrix[fbus,idx] = BR
        ctr_vr_matrix[tbus,idx] = DR

        ctr_vi_matrix[fbus,idx] = -BI
        ctr_vi_matrix[tbus,idx] = -DI

        cti_vr_matrix[fbus,idx] = BI
        cti_vr_matrix[tbus,idx] = DI

        cti_vi_matrix[fbus,idx] = BR
        cti_vi_matrix[tbus,idx] = DR

    cfr_hat = np.matmul(vr_hat, cfr_vr_matrix) + np.matmul(vi_hat, cfr_vi_matrix)
    cfi_hat = np.matmul(vr_hat, cfi_vr_matrix) + np.matmul(vi_hat, cfi_vi_matrix)
    ctr_hat = np.matmul(vr_hat, ctr_vr_matrix) + np.matmul(vi_hat, ctr_vi_matrix)
    cti_hat = np.matmul(vr_hat, cti_vr_matrix) + np.matmul(vi_hat, cti_vi_matrix)

    np.savetxt("System_data\cfr_vr_matrix.csv", cfr_vr_matrix, delimiter= ',')
    np.savetxt("System_data\cfi_vr_matrix.csv", cfi_vr_matrix, delimiter= ',')
    np.savetxt("System_data\ctr_vr_matrix.csv", ctr_vr_matrix, delimiter= ',')
    np.savetxt("System_data\cti_vr_matrix.csv", cti_vr_matrix, delimiter= ',')

    np.savetxt("System_data\cfr_vi_matrix.csv", cfr_vi_matrix, delimiter= ',')
    np.savetxt("System_data\cfi_vi_matrix.csv", cfi_vi_matrix, delimiter= ',')
    np.savetxt("System_data\ctr_vi_matrix.csv", ctr_vi_matrix, delimiter= ',')
    np.savetxt("System_data\cti_vi_matrix.csv", cti_vi_matrix, delimiter= ',')


    return cfr_hat, cfi_hat, ctr_hat, cti_hat 
    
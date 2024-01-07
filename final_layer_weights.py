import numpy as np
import pandas as pd


def final_layer_weights():
    # Get number of current measurements
    output_buses = np.arange(1,119)
    bdata = np.genfromtxt('System_data/bdata.csv', delimiter = ',')
    m_b = bdata.shape[0]*2
    m_s = output_buses.shape[0]
    m = m_s + m_b


    weights1 = np.zeros([m_b, m_b])
    weights2 = np.zeros([m_b, m_s])

    for i in range(weights1.shape[0]):
        weights1[i,i] = 1

    for i, bus in enumerate(output_buses):
        for j, branch in enumerate(bdata):
            if branch[0] == bus:
                weights2[j*2,i] = 1

            if branch[1] == bus:
                weights2[j*2+1,i] = 1




    weights = np.hstack((weights1, weights2))

    return weights

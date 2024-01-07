import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


def std_bins(std_p, n_bins =1):
    b_list = [[] for i in range(n_bins)]
    bin_limits = plt.hist(std_p, bins = n_bins)
    clear_output()
    for idx, std in enumerate(std_p):
        for j in range(len(bin_limits[1])):
            if std<= bin_limits[1][j]:
                if j==0:
                    b_list[0].append(idx)
                else:
                    b_list[j-1].append(idx)
                break
    
    no_load_buses = [5, 9, 30, 37, 38, 63, 64, 68, 71, 81]
    no_load_index = [i-1 for i in no_load_buses]

    for i in range(len(b_list)):
        red_list = [i for i in b_list[i] if i not in no_load_index]
        b_list[i] = red_list
    
    for bin in b_list:
        std_list = [std_p[i] for i in bin]
        print(np.std(std_list)*100/np.mean(std_list), '%')
    
    return b_list

def std_bins_bflow(std_b, n_bins =1):
    b_list = [[] for i in range(n_bins)]
    bin_limits = plt.hist(std_b, bins = n_bins)
    clear_output()
    for idx, std in enumerate(std_b):
        for j in range(len(bin_limits[1])):
            if std<= bin_limits[1][j]:
                if j==0:
                    b_list[0].append(idx)
                else:
                    b_list[j-1].append(idx)
                break

    
    for bin in b_list:
        std_list = [std_b[i] for i in bin]
        print(np.std(std_list)*100/np.mean(std_list), '%')
    
    return b_list
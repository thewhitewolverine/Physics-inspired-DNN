import pandas as pd
import numpy as np
from noise_models import *


def branch_flow_VI_split(bflow, VM, VA, CFM, CFA, CTM, CTA, Pinj, Qinj, num_bus = 11, add_bus = []):

	pmu_loc = {
		11: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81] ,
		13: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] ,
		32: [1, 5, 9, 12,13,17,21,23,26,28,34,37,41,45,49,53,56,62,63,68,71,75,77,80,85,86,90,94,101,105,110,114],
		118: np.arange(1,119)
	}

	pmu_bus = pmu_loc[num_bus]
	for bus in add_bus:
		pmu_bus.append(bus)
	pmu_index = [i-1 for i in pmu_bus]


	x_va = VA[pmu_index]
	x_vm = VM[pmu_index]

	bdata = np.genfromtxt('System_data/bdata.csv', delimiter = ',')

	cfm = CFM.values
	cfa = CFA.values

	ctm = CTM.values
	cta = CTA.values

	idx_from_dict = []
	idx_to_dict = []
	
	for branch in range(bdata.shape[0]):
		if int(bdata[branch,0]) not in pmu_bus:
			idx_from_dict.append(branch)
			
		if int(bdata[branch,1]) not in pmu_bus:
			idx_to_dict.append(branch)

	# print(len(idx_from_dict), len(idx_to_dict))
	cfm = np.delete(cfm, idx_from_dict, 1)
	cfa = np.delete(cfa, idx_from_dict, 1)

	ctm = np.delete(ctm, idx_to_dict, 1)
	cta = np.delete(cta, idx_to_dict, 1)

	CM = np.concatenate((cfm, ctm), axis=1)
	CA = np.concatenate((cfa, cta), axis=1)
	

	yt = [bflow]
	Y_tot = pd.concat(yt, axis=1)
	ytot = Y_tot.to_numpy()


	return x_vm.values, x_va.values, CM, CA, ytot

def Branch_flow_custom_xy_split(bflow, VM, VA, CFM, CFA, CTM, CTA, Pinj, Qinj, output = "vi_all",  num_bus = 11, add_bus = []):
	
	pmu_loc = {
		11: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81] ,
		13: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] ,
		32: [1, 5,9, 12,13,17,21,23,26,28,34,37,41,45,49,53,56,62,63,68,71,75,77,80,85,86,90,94,101,105,110,114],
		118: np.arange(1,119)
	}

	pmu_bus = pmu_loc[num_bus]

	for bus in add_bus:
		pmu_bus.append(bus)
	pmu_index = [i-1 for i in pmu_bus]


	x_va = VA[pmu_index]
	x_vm = VM[pmu_index]

	bdata = np.genfromtxt('System_data/bdata.csv', delimiter = ',')

	cfm = CFM.values
	cfa = CFA.values

	ctm = CTM.values
	cta = CTA.values

	idx_from_dict = []
	idx_to_dict = []
	
	for branch in range(bdata.shape[0]):
		if int(bdata[branch,0]) not in pmu_bus:
			idx_from_dict.append(branch)
			
		if int(bdata[branch,1]) not in pmu_bus:
			idx_to_dict.append(branch)

	# print(len(idx_from_dict), len(idx_to_dict))
	cfm = np.delete(cfm, idx_from_dict, 1)
	cfa = np.delete(cfa, idx_from_dict, 1)

	ctm = np.delete(ctm, idx_to_dict, 1)
	cta = np.delete(cta, idx_to_dict, 1)

	CM = np.concatenate((cfm, ctm), axis=1)
	CA = np.concatenate((cfa, cta), axis=1)
	
	if output == 'vi_all':
		yt = [VM, VA, CFM, CFA, CTM, CTA]
	elif output == 'powers':
		yt = [bflow, Pinj]
	elif output == 'all':
		yt = [VM, VA, CFM, CFA, CTM, CTA, bflow, Pinj]
	elif output == 'picdnn':
		yt = [bflow]

	Y_tot = pd.concat(yt, axis=1)
	ytot = Y_tot.to_numpy()

	
	return x_vm.values, x_va.values, CM, CA, ytot

def lin_st_est_VI(VM, VA, CFM, CFA, CTM, CTA, num_bus = 11, add_bus = []):

	pmu_loc = {
		11: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81] ,
		13: [8, 9, 10, 26, 30, 38, 63, 64, 65, 68, 81, 87, 111] ,
		32: [1, 5,9, 12,13,17,21,23,26,28,34,37,41,45,49,53,56,62,63,68,71,75,77,80,85,86,90,94,101,105,110,114],
		118: np.arange(1,119)
	}

	pmu_bus = pmu_loc[num_bus]
	for bus in add_bus:
		pmu_bus.append(bus)
	pmu_index = [i-1 for i in pmu_bus]


	x_va = VA[pmu_index]
	x_vm = VM[pmu_index]

	bdata = np.genfromtxt('bdata.csv', delimiter = ',')

	cfm = CFM.values
	cfa = CFA.values

	ctm = CTM.values
	cta = CTA.values

	idx_from_dict = []
	idx_to_dict = []
	
	for branch in range(bdata.shape[0]):
		if int(bdata[branch,0]) not in pmu_bus:
			idx_from_dict.append(branch)
			
		if int(bdata[branch,1]) not in pmu_bus:
			idx_to_dict.append(branch)

	print(len(idx_from_dict), len(idx_to_dict))
	cfm = np.delete(cfm, idx_from_dict, 1)
	cfa = np.delete(cfa, idx_from_dict, 1)

	ctm = np.delete(ctm, idx_to_dict, 1)
	cta = np.delete(cta, idx_to_dict, 1)


	return x_vm.values, x_va.values, cfm, cfa, ctm, cta

def feature_augment(pmu_bus):
	bdata = np.genfromtxt('bdata.csv', delimiter = ',')
	bbus = []
	for i in range(bdata.shape[0]):
		if bdata[i,0] in pmu_bus or bdata[i,1] in pmu_bus:
			bbus.append(int(bdata[i,0]))
			bbus.append(int(bdata[i,1]))
	
	return np.unique(bbus)
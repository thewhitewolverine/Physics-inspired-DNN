import pandas as pd

def import_antos_data():
	# Pinj = pd.read_csv('dataset_pf_corrected\P_injection.csv', header = None)
	# Qinj = pd.read_csv('dataset_pf_corrected\Q_injection.csv', header = None)
	# bflow = pd.read_csv('dataset_pf_corrected\B_flow_p.csv')
	VM = pd.read_csv('Antos_dataset/VDATA_mag_train_T1_12am.csv', header = None)
	VA = pd.read_csv('Antos_dataset/VDATA_ang_train_T1_12am.csv', header = None)
	CTM = pd.read_csv('Antos_dataset/It_mag_train_T1_12am.csv', header = None)
	CTA = pd.read_csv('Antos_dataset/It_ang_train_T1_12am.csv', header = None)
	CFM = pd.read_csv('Antos_dataset/If_mag_train_T1_12am.csv', header = None)
	CFA = pd.read_csv('Antos_dataset/If_ang_train_T1_12am.csv', header = None)
	
	return VM, VA, CFM, CFA, CTM, CTA

def import_branch_data():
	Pinj = pd.read_csv('dataset_pf_corrected/P_injection.csv', header = None)
	Qinj = pd.read_csv('dataset_pf_corrected/Q_injection.csv', header = None)
	bflow = pd.read_csv('dataset_pf_corrected/B_flow.csv', header = None, )
	VM = pd.read_csv('dataset_pf_corrected/VM_npu.csv', header = None)
	VA = pd.read_csv('dataset_pf_corrected/Volt_ang.csv', header = None)
	
	return VM, VA, bflow, Pinj, Qinj


def import_curr_branch_data():
	Pinj = pd.read_csv('dataset_pf_corrected/P_injection.csv', header = None)
	Qinj = pd.read_csv('dataset_pf_corrected/Q_injection.csv', header = None)
	bflow = pd.read_csv('dataset_pf_corrected/B_flow_p.csv')
	VM = pd.read_csv('dataset_pf_corrected/V_real.csv', header = None)
	VA = pd.read_csv('dataset_pf_corrected/V_imag.csv', header = None)
	CFM = pd.read_csv('dataset_pf_corrected/CF_real.csv', header = None)
	CFA = pd.read_csv('dataset_pf_corrected/CF_imag.csv', header = None)
	CTM = pd.read_csv('dataset_pf_corrected/CT_real.csv', header = None)
	CTA = pd.read_csv('dataset_pf_corrected/CT_imag.csv', header = None)
	
	return VM, VA, bflow, Pinj, Qinj, CFM, CFA, CTM, CTA
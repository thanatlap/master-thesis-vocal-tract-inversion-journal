'''
Modify (as of 05 Nov 2019)
- in randomize_by_percent_change function, the highest percent change is adjust from 1 to 0.5
  and the lowest percent change is 0.001
- remove randomize_by_weight_warping function
- in randomize_params function, add articulatory param 21, 22, 23 to have a fix value of -0.05 (MS parameters)
'''

import numpy as np
import math

def randomize_by_percent_change(predefine_params, from_idx, to_idx):
	
	percent_change = np.random.uniform(0.001, high=0.5, size=1)[0]
	return [((predefine_params[to_idx][i] - predefine_params[from_idx][i])*percent_change) + predefine_params[from_idx][i] for i in  range(predefine_params.shape[1])]

def randomize_params(predefine_params, param_high, param_low, sampling_step=0.025):

	'''
	Same as randomize_based_on_predefine_params() but the formula to calculate TRX and TRY are from 
	JD.speaker
	'''
	rand_param = []
	# find min and max along row
	min_param = np.min(predefine_params, axis=0)
	max_param = np.max(predefine_params, axis=0)

	for j, _ in enumerate(param_high):
		# if the parameter is WC, automatically set to 0.0
		if j == 8:
			rand_param.append(0.0)
		# auto calculate TRX from TCY
		elif j == 15:
			rand_param.append(rand_param[10]*0.9380 - 5.1100)
		# calculate TRY from TCX
		elif j == 16:
			rand_param.append(rand_param[9]*0.8310 -3.0300)
		elif j in [21,22,23]:
			rand_param.append(-0.05) 
		else:
			rand_param.append(np.random.choice(np.arange(min_param[j], max_param[j], step=sampling_step)))

	return rand_param
import numpy as np
import math, itertools



def randomly_select_syllable_pair(predefined_syllables, syllable_labels, is_disyllable, warp_constraint):

	# random select syllable pair  
	num_item_pair = 4 if is_disyllable else 2
	while True:
		syllable_pair = [ np.random.randint(predefined_syllables.shape[0]) for i in range(num_item_pair)]
		# check duplicate index
		if ((len(syllable_pair) - len(set(syllable_pair))) == 0):
			break
	while True:
		random_syllables = [random_param_from_syllable_pair(syllable_pair[idx*2], syllable_pair[idx*2+1], predefined_syllables, warp_constraint) for idx in range(int(num_item_pair/2))]
		random_phonetic = [syllable_labels[syllable_pair[idx*2]] for idx in range(int(num_item_pair/2))]

		# Check if the param is directly pick from default param since we will used this as a final testing set.
		if sum([0 if not (item in predefined_syllables.tolist()) else 1 for item in random_syllables]) == 0:
			break
	# If generate disyllable, return a pair of param else, return only first pair
	random_syllables = random_syllables if is_disyllable else random_syllables[0]
	random_phonetic = random_phonetic if is_disyllable else random_phonetic[0]
	return random_syllables, random_phonetic

def random_param_from_syllable_pair(from_idx, to_idx, predefined_syllables, warp_constraint):
	return randomize_by_percent_change(predefined_syllables, from_idx, to_idx, warp_constraint[0], warp_constraint[1])

def randomize_by_percent_change(predefine_params, from_idx, to_idx, min_percent, max_percent):
	return [((predefine_params[to_idx][i] - predefine_params[from_idx][i])*np.random.uniform(min_percent, high=max_percent)) + predefine_params[from_idx][i] for i in  range(predefine_params.shape[1])]

def check_duplicate_and_remove(random_param, total_aggregate_param, random_phonetic):

	duplicate_count = 0
	for idx, item in enumerate(random_param):
		temp = random_param.copy()
		_ = temp.pop(idx)
		if (item in temp) or (item in total_aggregate_param):
			_ = random_param.pop(idx)
			_ = random_phonetic.pop(idx)
			duplicate_count += 1

	return random_param, duplicate_count, random_phonetic
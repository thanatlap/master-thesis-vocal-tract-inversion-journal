import numpy as np
import math, itertools

def random_param_from_syllable_pair(from_idx, to_idx, predefined_syllables, warp_constraint):
	random_param = randomize_by_percent_change(predefined_syllables, from_idx, to_idx, warp_constraint[0], warp_constraint[1])
	return random_param

def randomly_select_syllable_pair(predefined_syllables, is_disyllable, warp_constraint):

	# random select syllable pair  
	num_item_pair = 4 if is_disyllable else 2
	while True:
		syllable_pair = [ np.random.randint(predefined_syllables.shape[0]) for i in range(num_item_pair)]
		# check duplicate index
		if ((len(syllable_pair) - len(set(syllable_pair))) == 0):
			break
	while True:
		random_syllables = [random_param_from_syllable_pair(syllable_pair[idx*2], syllable_pair[idx*2+1], predefined_syllables, warp_constraint) for idx in range(int(num_item_pair/2))]
		# Check if the param is directly pick from default param since we will used this as a final testing set.
		if sum([0 if not (item in predefined_syllables.tolist()) else 1 for item in random_syllables]) == 0:
			break
	# If generate disyllable, return a pair of param else, return only first pair
	return random_syllables if is_disyllable else random_syllables[0]

def randomize_by_percent_change(predefine_params, from_idx, to_idx, min_percent, max_percent):
	return [((predefine_params[to_idx][i] - predefine_params[from_idx][i])*np.random.uniform(min_percent, high=max_percent)) + predefine_params[from_idx][i] for i in  range(predefine_params.shape[1])]

def check_duplicate_and_remove(random_param, total_aggregate_param):
	# sort list
	random_param.sort()
	# group by item, if item in it list is duplicated, it will be group into (select only) one item.
	new_random_param = list(item for item,_ in itertools.groupby(random_param))
	# calculated the number of duplication item
	duplicate_item_count = int(len(random_param) - len(new_random_param))
	# check if the newly random params is duplicated with existing random params
	if total_aggregate_param != []:
		for item in new_random_param:
			if item in total_aggregate_param:
				new_random_param.remove(item)
				duplicate_item_count += 1

	return new_random_param, duplicate_item_count
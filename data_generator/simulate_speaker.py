'''
Modify (as of 05 Nov 2019)
- in gesture generation function, the duration of first syllable is change from random from given list of
  [0.5, 0.6, 0.7] to uniform random between 0.4 to 0.8
- Adjust f0 high and low value of each duration
- Adjust first syllable duration
'''

import numpy as np
import re
import pandas as pd
import os
from os.path import join
import config as cf
from functools import partial

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# predefine param
adult_high = np.array([1, -3.5, 0, 0, 1, 4, 1, 1, 1, 4, 1, 5.5, 2.5, 4, 5, 2, 0, 1.4, 1.4, 1.4, 1.4, 0.3, 0.3, 0.3])
adult_low = np.array([0,-6.0, -0.5, -7.0, -1.0, -2.0, 0, -0.1, 0, -3, -3, 1.5, -3.0, -3, -3, -4, -6, -1.4, -1.4, -1.4, -1.4, -0.05, -0.05, -0.05]) 
adult_neutral = np.array([1.,-4.75,0.,-2.,-0.07,0.95,0.,-0.1,0.,-0.4,-1.46,3.5,-1.,2.,0.5,0.,0.,0.,0.06,0.15,0.15,-0.05,-0.05,-0.05])

infant_high = np.array([1., -1.85, 0., 0., 1., 2.205, 1., 1., 1.,2.327, 0.630, 3.200, 1.457, 2.327, 2.835, 1.163, 0.079, 1.081, 1.081, 1.081, 1.081, 0.3, 0.3, 0.3])
infant_low = np.array([0., -3.228, -0.372, -7., -1., -1.102, 0.,-0.1, 0., -3.194, -1.574, 0.873, -1.574, -3.194, -1.574, -4.259, -3.228, -1.081, -1.081, -1.081, -1.081, 0., 0., 0.])
infant_neutral = np.array([1.,-2.643,0.,-2.,-0.07,0.524,0.,-0.1,0.,-0.426,-0.767,2.036,-0.578,1.163,0.321,0.,0.053,0.,0.046,0.116,0.116,0.,0.,0.])

adult_narrow=np.array([1.8,   0, 1.05,   -0.2,   1.55, -1.2,  2.68,  -3.2, 1.48,  -3.2, 1.1,  -1.2,  0, -1, 0, 0])
infant_narrow=np.array([1.378, 0 ,0.8036, -0.125, 1.08, -0.75, 1.768, -2,   1.133, -2,   0.63, -0.75, 0, -0.625, 0, 0])
adult_wide=np.array([3.3, 0, 2.13, -0.2, 1.9, -1.2, 2.68, -3.2, 1.48, -3.2, 1.45,  -1.2,  0, -1, 0, 0])
infant_wide=np.array([2.526, 0, 1.63, -0.125, 1.348, -0.75, 1.768, -2, 1.133, -2, 0.8978, -0.75, 0, -0.625, 0, 0])

param_name = np.array(["HX","HY","JX","JA","LP","LD","VS","VO","WC","TCX","TCY","TTX","TTY","TBX","TBY","TRX","TRY","TS1","TS2","TS3","TS4","MA1","MA2","MA3"])

def _extract_param(speaker_path):
	# extract the numeric value from text file
	list_param = []
	with open(speaker_path, 'r') as file:
		text = [line.rstrip('\n') for line in file]   
	for content in text:
		numeric = re.findall(r"[-+]?\d*\.\d+|\d+", content)
		numeric = [float(i) for i in numeric]
		if numeric:
			list_param.append(np.array(numeric))
	return np.array(list_param)

def simulate_speaker(scale_by, sid):
	'''
	simulated speaker by scaling between adult to infant
	'''

	adult_param = _extract_param(cf.ADULT_SPEAKER_FILE)
	infant_param = _extract_param(cf.INFANT_SPEAKER_FILE)

	# compute the different of the parameter value between adult and infant
	# then, compute the change from the adult param to get new_param
	new_vocaltract_param = []
	for idx, param in enumerate(adult_param):
		new_vocaltract_param.append(param - (param - infant_param[idx])*np.full(param.shape, fill_value=scale_by))

	# compute a change of min, max, and neutral value of each vocaltract parameters
	change_high = adult_high - infant_high
	change_low = adult_low - infant_low
	change_neutral = adult_neutral - infant_neutral

	# decrease sizing simulation of a new high and low
	new_high = adult_high - (np.full(change_high.shape, fill_value=scale_by)*change_high)
	new_low = adult_low - (np.full(change_low.shape, fill_value=scale_by)*change_low)
	new_neutral = adult_neutral - (np.full(change_neutral.shape, fill_value=scale_by)*change_neutral)

	# compute larynx narrow and wide points
	change_narrow = np.nan_to_num(1-(infant_narrow/(adult_narrow)))
	change_wide = np.nan_to_num(1-(infant_wide/(adult_wide)))

	new_narrow = adult_narrow - adult_narrow*(np.full(change_narrow.shape, fill_value=scale_by)*change_narrow)
	new_wide = adult_wide - adult_wide*(np.full(change_wide.shape, fill_value=scale_by)*change_wide)

	os.makedirs(cf.SPKEAKER_SIM_DIR, exist_ok = True)
	np.savez(join(cf.SPKEAKER_SIM_DIR,'speaker_param%s.npz'%sid), high=new_high, low=new_low, neutral=new_neutral)
	filename = join(cf.SPKEAKER_SIM_DIR,'speaker_s%s.speaker'%sid)

	def _template():
		f.write('<speaker>\n<vocal_tract_model>\n<anatomy>\n<palate>\n')
		for i in range(9):
			f.write('<p%s x="%s" z="%s" teeth_height="%s" top_teeth_width="%s" bottom_teeth_width="%s" palate_height="%s" palate_angle_deg="%s"/>\n'%
					(i,new_vocaltract_param[i][1],new_vocaltract_param[i][2],new_vocaltract_param[i][3],new_vocaltract_param[i][4],new_vocaltract_param[i][5],new_vocaltract_param[i][6],new_vocaltract_param[i][7]))
		f.write('</palate>\n')
		f.write('<jaw fulcrum_x="%s" fulcrum_y="%s" rest_pos_x="%s" rest_pos_y="%s" tooth_root_length="%s">\n'%
			   (new_vocaltract_param[9][0],new_vocaltract_param[9][1],new_vocaltract_param[9][2],new_vocaltract_param[9][3],new_vocaltract_param[9][4]))
		for i in range(9):
			f.write('<p%s x="%s" z="%s" teeth_height="%s" top_teeth_width="%s" bottom_teeth_width="%s" jaw_height="%s" jaw_angle_deg="%s"/>\n'%
					(i,new_vocaltract_param[i+10][1],new_vocaltract_param[i+10][2],new_vocaltract_param[i+10][3],new_vocaltract_param[i+10][4],new_vocaltract_param[i+10][5],new_vocaltract_param[i+10][6],new_vocaltract_param[i+10][7]))
		f.write('</jaw>\n<lips width="%s"/>\n<tongue>\n'%new_vocaltract_param[19][0])
		f.write('<tip radius="%s"/>\n'%new_vocaltract_param[20][0])
		f.write('<body radius_x="%s" radius_y="%s"/>\n'%(new_vocaltract_param[21][0],new_vocaltract_param[21][1]))
		f.write('<root automatic_calc="0" trx_slope="%s" trx_intercept="%s" try_slope="%s" try_intercept="%s"/>\n </tongue>\n'%(
			new_vocaltract_param[22][1],new_vocaltract_param[22][2],new_vocaltract_param[22][3],new_vocaltract_param[22][4]))
		f.write('<velum uvula_width="%s" uvula_height="%s" uvula_depth="%s" max_nasal_port_area="%s" >\n'%(
			new_vocaltract_param[23][0],new_vocaltract_param[23][1],new_vocaltract_param[23][2],new_vocaltract_param[23][3]))
		f.write('<low points="')
		for i in range(10):
			f.write('%s '%new_vocaltract_param[24][i])
		f.write('"/>\n')
		f.write('<mid points="')
		for i in range(10):
			f.write('%s '%new_vocaltract_param[25][i])
		f.write('"/>\n')
		f.write('<high points="')
		for i in range(10):
			f.write('%s '%new_vocaltract_param[26][i])
		f.write('"/>\n')
		f.write('</velum>\n')
		f.write('<pharynx fulcrum_x="%s" fulcrum_y="%s" rotation_angle_deg="%s" top_rib_y="%s" upper_depth="%s" lower_depth="%s" back_side_width="%s"/>'%(
		new_vocaltract_param[27][0],new_vocaltract_param[27][1],new_vocaltract_param[27][2],new_vocaltract_param[27][3],new_vocaltract_param[27][4],new_vocaltract_param[27][5],new_vocaltract_param[27][6]))
		f.write('<larynx upper_depth="%s" lower_depth="%s" epiglottis_width="%s" epiglottis_height="%s" epiglottis_depth="%s" epiglottis_angle_deg="%s">'%(new_vocaltract_param[28][0],new_vocaltract_param[28][1],new_vocaltract_param[28][2],new_vocaltract_param[28][3],new_vocaltract_param[28][4],new_vocaltract_param[28][5]))
		f.write('<narrow points="')
		for i in range(16):
			f.write('%s '%new_narrow[i])
		f.write('"/>\n')
		f.write('<wide points="')
		for i in range(16):
			f.write('%s '%new_wide[i])
		f.write('"/>\n')
		f.write('</larynx>\n')
		f.write('<piriform_fossa length="%s" volume="%s"/>\n'%(new_vocaltract_param[31][0],new_vocaltract_param[31][1]))
		f.write('<subglottal_cavity length="%s"/>\n'%(new_vocaltract_param[32][0]))
		f.write('<nasal_cavity length="%s"/>\n'%(new_vocaltract_param[33][0]))
		for i in range(24):
			f.write('<param index="%s"  name="%s"  min="%s"  max="%s"  neutral="%s" />\n'%
					(i, param_name[i], new_low[i], new_high[i], new_neutral[i]))
		f.write('</anatomy>\n<shapes>\n')

	f = open(filename, 'w')
	_template()
	f.close()

	# simulate a full speaker vocaltract model
	default_param = pd.read_csv(cf.PREDEFINE_PARAM_FILE).values
	scale_default_param = (default_param - adult_low)/(adult_high - adult_low)
	new_param = scale_default_param*(new_high - new_low) + new_low
	with open(cf.LABEL_NAME) as f:
		syllable = np.array([word.strip() for line in f for word in line.split(',')])

	filename = join(cf.SPKEAKER_SIM_DIR,'speaker_s%s_full.speaker'%sid)
	f = open(filename, 'w')
	_template()
	for i, syl in enumerate(syllable):
		f.write('<shape name="%s">\n'%syl)
		for j, name in enumerate(param_name):
			f.write('<param name="%s" value="%.4f"/>\n'%(name, new_param[i][j]))
		f.write('</shape>\n')
	f.write(open(cf.TAIL_SPEAKER,'r').read())
	f.close()

def create_speaker(filename, params, speaker_head, speaker_tail, is_disyllable):
	'''
	create speaker file using simulated speaker vocaltract
	'''

	f = open(filename, 'w')
	f.write(speaker_head)
	if is_disyllable:

		# Loop through each parameter in the list
		for idx, pair_param in enumerate(params):
			if idx == 0: 
				f.write('<shape name="aaa">\n')
			else:
				f.write('<shape name="bbb">\n')
			for jdx, param in enumerate(pair_param):
				f.write('<param name="%s" value="%.2f"/>\n'%(param_name[jdx],param))
			f.write('</shape>\n')

	else:
		f.write('<shape name="aaa">\n')
		for jdx, param in enumerate(params):
			f.write('<param name="%s" value="%.2f"/>\n'%(param_name[jdx],param))
		f.write('</shape>\n')

	# Close with tail part of the speaker file
	f.write(speaker_tail)
	f.close()

def create_ges(ges_file, is_disyllable):
	'''
	randomize f0-gesture to simulate different speaker style
	'''
	ges_head = open(cf.GES_HEAD,'r').read()

	# max duration of the entire speech sound
	max_duration = np.random.uniform(low=1.0, high=1.5) if is_disyllable else np.random.uniform(low=0.4, high=0.7)
	
	# if disyllable, the duration of fisrt and second syllabic are vary
	a_duration = np.random.uniform(0.45*max_duration, high=0.55*max_duration) if is_disyllable else max_duration
	b_duration = max_duration - a_duration if is_disyllable else None # b_duration is not used for monosyllable

	# time constant (The VTL constrict value to 0.04, inspect from empirical study (manually play with program))
	a_time_constant = np.random.uniform(0.015, high=0.02)
	b_time_constant = np.random.uniform(0.015, high=0.02)

	# f0 gesture duration
	total_duration = max_duration
	duration = []
	# first f0 gesture duration is small
	duration.append(np.random.choice([0.01,0.05]))
	total_duration -= duration[0]
	# first f0 gesture duration is longer or until max
	duration.append(np.random.uniform(low=0.45*total_duration, high=0.65*total_duration))
	total_duration -= duration[1]
	# if max duration is less than 50ms then f0-gesture only have 2 phase, else, the rest
	duration.append(total_duration)
		

	total_duration -= duration[1]
		
		
	# create gesture file
	f = open(ges_file, 'w')
	f.write('<gestural_score>\n<gesture_sequence type="vowel-gestures" unit="">\n>')
	f.write('<gesture value="aaa" slope="0.000000" duration_s="%.6f" time_constant_s="%.6f" neutral="0" />\n'%(a_duration, a_time_constant))
	if is_disyllable:
		f.write('<gesture value="bbb" slope="0.000000" duration_s="%.6f" time_constant_s="%.6f" neutral="0" />\n'%(b_duration,b_time_constant))
	f.write(ges_head)
	f.write('<gesture value="modal" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%a_duration)
	if is_disyllable:
		f.write('<gesture value="modal" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%b_duration)
	f.write('</gesture_sequence>\n')
	f.write('<gesture_sequence type="f0-gestures" unit="st">\n')
	for idx, d in enumerate(duration):
		if idx == 0:
			f0 = np.random.uniform(low=80, high=81)
		elif idx == 1:
			f0 = np.random.uniform(low=81, high=83)
		else:
			f0 = f0
		f.write('<gesture value="%.5f" slope="0.000000" duration_s="%.5f" time_constant_s="0.030000" neutral="0"/>\n'%(f0,d))
	f.write('</gesture_sequence>\n')
	f.write('<gesture_sequence type="lung-pressure-gestures" unit="dPa">\n')
	f.write('<gesture value="0.000000" slope="0.000000" duration_s="0.010000" time_constant_s="0.005000" neutral="0" />\n')
	f.write('<gesture value="8000.000000" slope="0.000000" duration_s="%.6f" time_constant_s="0.005000" neutral="0" />\n'%a_duration)
	if is_disyllable:
		f.write('<gesture value="8000.000000" slope="0.000000" duration_s="%.6f" time_constant_s="0.005000" neutral="0" />\n'%b_duration)
	f.write('</gesture_sequence> \n')
	f.write('</gestural_score>\n')
	f.close()

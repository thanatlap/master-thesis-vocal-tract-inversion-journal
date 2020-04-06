import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from os.path import join
import scipy 
from scipy.stats import ttest_ind
import matplotlib.patches as mpatches

def sep_label_position(df):
	df[['Label','Position']] = df['Label'].str.split(';', expand=True) 
	return df

def prep_df(df, is_disyllable):
	if is_disyllable:
		df = sep_label_position(df)
	return df

def custome_reindex(df):
	return df.reindex(["a:", "i:", "u:","e:","E","@","9","o","O"])

def custome_reindex_type2(df):
	return df.reindex(["a:", "i:", "u:","e:",'ɛ:','ə:','œ:','o:','ɔ:', 'ɑ:','ø:','ʊ:'])

def custome_reindex_type3(df):
	return df.reindex(["a:", "i:", "u:","e:",'ɛ:','ɯ:','ɤ:','o:','ɔ:'])

def compute_formant_rmse(formant_df, dir_path, experiment_num, is_disyllable, label_set):

	formant_df = set_datapoint_index(formant_df)

	if label_set == 1:
		formant_df = change_label_set_1(formant_df)
	else:
		formant_df = change_label_set_2(formant_df)

	if is_disyllable:
		formant_rmse = formant_df.groupby(['Label', 'Position']).mean().pivot_table(index=['Label'],columns='Position',values=['F1','F2','F3'],fill_value=0)
		formant_rmse.columns = ['-'.join(col).strip() for col in formant_rmse.columns.values]
	else:
		formant_rmse = formant_df.copy().set_index('Label')

	if label_set == 1:
		formant_rmse = custome_reindex_type2(formant_rmse)
	else:
		formant_rmse = custome_reindex_type3(formant_rmse)

	formant_rmse.to_csv(join(dir_path,'formant_rmse_exp'+str(experiment_num)+'.csv'))

def compute_formant_ttest(datapoint_df, dir_path, experiment_num, is_disyllable):
	
	datapoint_prep = datapoint_df.pivot_table(index=['Label'],columns='Target',values=['F1','F2','F3'],fill_value=0)
	datapoint_prep.columns = [(str(col[0])+'-'+str(col[1])).strip() for col in datapoint_prep.columns.values]
	datapoint_prep = prep_df(datapoint_prep.reset_index(), is_disyllable)

	if is_disyllable:
		first_syl = datapoint_prep[datapoint_prep['Position']=='1']
		second_syl = datapoint_prep[datapoint_prep['Position']=='2']
		
		f_f1 = ttest_ind(first_syl['F1-0'], first_syl['F1-1'])
		f_f2 = ttest_ind(first_syl['F2-0'], first_syl['F2-1'])
		f_f3 = ttest_ind(first_syl['F3-0'], first_syl['F3-1'])
		s_f1 = ttest_ind(second_syl['F1-0'], second_syl['F1-1'])
		s_f2 = ttest_ind(second_syl['F2-0'], second_syl['F2-1'])
		s_f3 = ttest_ind(second_syl['F3-0'], second_syl['F3-1'])
		
		ttest_df = pd.DataFrame({'result':['t-test','p-value'], 
								 '1F1':[f_f1[0], f_f1[1]],
								 '1F2':[f_f2[0], f_f2[1]],
								 '1F3':[f_f3[0], f_f3[1]],
								 '2F1':[s_f1[0], s_f1[1]],
								 '2F2':[s_f2[0], s_f2[1]],
								 '2F3':[s_f3[0], s_f3[1]]})
	else:

		f_f1 = ttest_ind(datapoint_prep['F1-0'], datapoint_prep['F1-1'])
		f_f2 = ttest_ind(datapoint_prep['F2-0'], datapoint_prep['F2-1'])
		f_f3 = ttest_ind(datapoint_prep['F3-0'], datapoint_prep['F3-1'])
		
		ttest_df = pd.DataFrame({'result':['t-test','p-value'], 
								 '1F1':[f_f1[0], f_f1[1]],
								 '1F2':[f_f2[0], f_f2[1]],
								 '1F3':[f_f3[0], f_f3[1]]})
 
	ttest_df.to_csv(join(dir_path,'ttest_exp'+str(experiment_num)+'.csv'))

def set_datapoint_index(df):
	df_temp = df.copy()
	df_temp['Label_idx'] = df_temp['Label']
	return df_temp.set_index('Label_idx')

def plot_formant_chart(df, reindex_fn, dir_path, is_disyllable, note=None):

	filled_markers = ['o', 'v', "s", "P", "*", "D", "^", "X", "<", "d", "x", "+"]

	actual = reindex_fn(df[df['Target']==1].copy())
	# filled_markers = actual.index
	actual = actual.reset_index(drop=True)
	estimated = reindex_fn(df[df['Target']==0].copy()).reset_index(drop=True)

	fig, ax = plt.subplots()

	for idx, data in enumerate(actual.index):
		mark = filled_markers[idx]
		# mark = '${}$'.format(filled_markers[idx])
		# s=150
		# linewidth='0.5'
		ax.scatter(actual['F2'][idx], actual['F1'][idx], marker=mark,  color='red', label=actual['Label'][idx])
		ax.scatter(estimated['F2'][idx], estimated['F1'][idx], marker=mark, color='blue')

	ax.set_xticks(np.arange(400, 2400+1, 200))
	ax.set_yticks(np.arange(100, 800+1, 100))
	
	plt.gca().invert_xaxis()
	plt.gca().invert_yaxis()

	ax.set_xlabel('F2 [Hz]', fontsize=14)
	ax.set_ylabel('F1 [Hz]', fontsize=14)

	ax.legend()
	ax.legend(bbox_to_anchor=(1.05, 0.8), loc=2, borderaxespad=0.)

	# ax.grid()

	leg = ax.get_legend()
	for idx,_ in enumerate(leg.legendHandles):
		leg.legendHandles[idx].set_color('black')

	red_patch = mpatches.Patch(color='red', label='Target')
	blue_patch = mpatches.Patch(color='blue', label='Estimated')
	legend1 = ax.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	plt.gca().add_artist(leg)
	plt.gcf().set_size_inches(7,5)

	syllable_type = 'disyllable' if is_disyllable else 'monosyllable'
	
	if note and is_disyllable:
		ax.set_title('Formant Chart of '+note+' ['+syllable_type+']')
		filename = 'formant_chart_'+syllable_type+' '+note
	else:
		ax.set_title('Formant Chart of ['+syllable_type+']')
		filename = 'formant_chart_'+syllable_type

	for file_type in ['.png','.pdf']:
		plt.savefig(join(dir_path, filename+file_type), dpi=300, bbox_extra_artists=(leg,legend1), bbox_inches='tight')
		
	plt.clf()


def change_label_set_1(datapoint_df):
	datapoint_df.at['E', 'Label']= 'ɛ:' 
	datapoint_df.at['O', 'Label']= 'ɔ:'
	datapoint_df.at['9', 'Label']= 'œ:'
	datapoint_df.at['@', 'Label']= 'ə:'
	datapoint_df.at['o', 'Label']= 'o:'
	datapoint_df.at['a', 'Label']= 'a:'
	datapoint_df.at['i', 'Label']= 'i:'
	datapoint_df.at['e', 'Label']= 'e:'
	datapoint_df.at['u', 'Label']= 'u:'
	datapoint_df.at['A', 'Label']= 'ɑ:'
	datapoint_df.at['2', 'Label']= 'ø:'
	datapoint_df.at['U', 'Label']= 'ʊ:'

	return datapoint_df

def change_label_set_2(datapoint_df):
	datapoint_df.at['E', 'Label']= 'ɛ:' 
	datapoint_df.at['O', 'Label']= 'ɔ:'
	datapoint_df.at['7', 'Label']= 'ɤ:'
	datapoint_df.at['M', 'Label']= 'ɯ:'
	datapoint_df.at['o', 'Label']= 'o:'
	datapoint_df.at['a', 'Label']= 'a:'
	datapoint_df.at['i', 'Label']= 'i:'
	datapoint_df.at['e', 'Label']= 'e:'
	datapoint_df.at['u', 'Label']= 'u:'
	return datapoint_df

def formant_chart(datapoint_df,  dir_path, experiment_num, is_disyllable, label_set):

	datapoint_df = prep_df(datapoint_df, is_disyllable)
	datapoint_df = set_datapoint_index(datapoint_df)

	if label_set == 1:
		datapoint_df = change_label_set_1(datapoint_df) 
	elif label_set == 2: 
		datapoint_df = change_label_set_2(datapoint_df) 

	if is_disyllable:
		datapoint_df = set_datapoint_index(datapoint_df.groupby(['Label','Target','Position']).mean().reset_index())
		first_syl = datapoint_df[datapoint_df['Position']=='1']
		second_syl = datapoint_df[datapoint_df['Position']=='2']
		if label_set == 1:
			plot_formant_chart(first_syl, custome_reindex_type2, dir_path, is_disyllable, note='FirstSyllable')
			plot_formant_chart(second_syl, custome_reindex_type2, dir_path, is_disyllable, note='SecondSyllable')
		elif label_set == 2: 
			plot_formant_chart(first_syl, custome_reindex_type3, dir_path, is_disyllable, note='FirstSyllable')
			plot_formant_chart(second_syl, custome_reindex_type3, dir_path, is_disyllable, note='SecondSyllable')

	else:
		datapoint_df = set_datapoint_index(datapoint_df)
		plot_formant_chart(datapoint_df, custome_reindex_type2, dir_path, is_disyllable)

def plot_each_syllable_formant(input_df, position, dir_path):
	

	syl_df = input_df[input_df['Position'] == str(position)]
	unique_labels = set(syl_df['Label'].copy())
	
	if len(unique_labels) == 9:
		fig, ax = plt.subplots(3,3, figsize=(20,15))
		unique_labels = ["a:", "i:", "u:","e:",'ɛ:','ɯ:','ɤ:','o:','ɔ:']
	elif len(unique_labels) == 12:
		fig, ax = plt.subplots(4,3, figsize=(20,25))
	else:
		fig, ax = plt.subplots(5,5, figsize=(20,15))

	colors = ['blue', 'red']
	target_label = ['estimated','original']


	for idx, label in enumerate(unique_labels):
		axs = ax[idx//3][idx%3]
		for jdx, target in enumerate([0.0, 1.0]):
			df = syl_df[syl_df['Target'] == target]
			axs.scatter(df[df['Label'] == label]['F2'], df[df['Label'] == label]['F1'], c=colors[jdx], label=target_label[jdx])
		axs.title.set_text(label)
		axs.set_xticks(np.arange(400, 2400+1, 200))
		axs.set_yticks(np.arange(100, 800+1, 100))
		axs.set_xlabel('F2')
		axs.set_ylabel('F1')
		axs.invert_xaxis()
		axs.invert_yaxis()
		axs.legend()
		
	for file_type in ['.png','.pdf']:
		plt.savefig(join(dir_path, 'formant_pos_'+str(position)+'_each'+file_type), dpi=300)

def formant_each_syllable_chart(datapoint_df,  dir_path, is_disyllable, label_set):

	datapoint_df = prep_df(datapoint_df, is_disyllable)
	datapoint_df = set_datapoint_index(datapoint_df)

	if label_set == 1:
		datapoint_df = change_label_set_1(datapoint_df)
	elif label_set == 2: 
		datapoint_df = change_label_set_2(datapoint_df) 

	if is_disyllable:
		plot_each_syllable_formant(datapoint_df, 1, dir_path)
		plot_each_syllable_formant(datapoint_df, 2, dir_path)
	else:
		datapoint_df['Position'] = '1'
		plot_each_syllable_formant(datapoint_df, 1, dir_path)

def generate_eval_result(experiment_num, is_disyllable, mode='eval', label_set=1, output_path=None):

	if label_set not in [1,2]:
		raise ValueError('Label set %s not valid! [1:eval, 2:predict] (16 Jan 2020)'%str(label_set)) 

	dir_path = 'result/'+mode+'_'+str(experiment_num)+'/formant/' if output_path==None else output_path

	print('[INFO] read data')
	formant_df = pd.read_csv(join(dir_path, 'formant_df.csv'))
	datapoint_df = pd.read_csv(join(dir_path,'data_point.csv'))

	formant_df = prep_df(formant_df, is_disyllable)

	print('[INFO] compute formant rmse')
	compute_formant_rmse(formant_df, dir_path, experiment_num, is_disyllable, label_set)
	print('[INFO] compute t-test')
	compute_formant_ttest(datapoint_df, dir_path, experiment_num, is_disyllable)
	print('[INFO] generate formant chart')
	formant_chart(datapoint_df,  dir_path, experiment_num, is_disyllable, label_set)
	print('[INFO] generate formant chart for each syllable')
	datapoint_df = pd.read_csv(join(dir_path,'data_point.csv'))
	formant_each_syllable_chart(datapoint_df,  dir_path, is_disyllable, label_set)
	
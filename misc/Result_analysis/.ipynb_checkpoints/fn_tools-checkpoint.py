import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from os.path import join
import scipy 
from scipy.stats import ttest_ind

def numpy_fillna(data):
    lens = np.array([len(i) for i in data])
    mask = np.arange(lens.max()) < lens[:,None]
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def cut_head_and_tail(data):
    return np.array([ item[int(0.25*item.shape[0]):int(0.75*item.shape[0])] for item in data])

def prep_formant_data(data):
    data = cut_head_and_tail(numpy_fillna(data))
    return data.reshape((data.shape[0]*data.shape[1])).astype('float')


def compute_formant_result_table(exp_list):
    res = []
    model = []
    f1e = []
    f2e = []
    f3e = []
    for exp_num in exp_list:

        path = '../../experiment/result/eval_{}/formant/'.format(exp_num)

        a1 = np.load(join(path, 'actual_F1.npy'))
        a2 = np.load(join(path, 'actual_F2.npy'))
        a3 = np.load(join(path, 'actual_F3.npy'))
        e1 = np.load(join(path, 'estimated_F1.npy'))
        e2 = np.load(join(path, 'estimated_F2.npy'))
        e3 = np.load(join(path, 'estimated_F3.npy'))

        a1 = prep_formant_data(a1)
        a2 = prep_formant_data(a2)
        a3 = prep_formant_data(a3)

        e1 = prep_formant_data(e1)
        e2 = prep_formant_data(e2)
        e3 = prep_formant_data(e3)

        f1_error = np.divide(np.absolute(a1 - e1), a1, out=np.zeros_like(a1), where=a1!=0.0)*100
        f2_error = np.divide(np.absolute(a2 - e2), a1, out=np.zeros_like(a2), where=a2!=0.0)*100
        f3_error = np.divide(np.absolute(a3 - e3), a1, out=np.zeros_like(a3), where=a3!=0.0)*100

        f11_error = np.mean(f1_error[f1_error > 0])
        f12_error = np.mean(f2_error[f2_error > 0])
        f13_error = np.mean(f3_error[f3_error > 0])

        f11_sd = np.std(f1_error)
        f12_sd = np.std(f2_error)
        f13_sd = np.std(f3_error)

        f11_sem = scipy.stats.sem(f1_error, axis=None)
        f12_sem = scipy.stats.sem(f2_error, axis=None)
        f13_sem = scipy.stats.sem(f3_error, axis=None)

        f11_cf = f11_sem * scipy.stats.t.ppf((1 + 0.95) / 2., f1_error.shape[0]-1)
        f12_cf = f12_sem * scipy.stats.t.ppf((1 + 0.95) / 2., f2_error.shape[0]-1)
        f13_cf = f13_sem * scipy.stats.t.ppf((1 + 0.95) / 2., f3_error.shape[0]-1)

        res.append([exp_num, f11_error, f12_error, f13_error, f11_sd, f12_sd, f13_sd, f11_cf, f12_cf, f13_cf])
        model.extend([exp_num]*f1_error.shape[0])
        f1e.extend(f1_error)
        f2e.extend(f2_error)
        f3e.extend(f3_error)

    df_res = pd.DataFrame(np.array(res),columns=['MODEL', 'F1 ERROR','F2 ERROR', 'F3 ERROR', 'F1 SD', 'F2 SD', 'F3 SD', 'F1 CF', 'F2 CF', 'F3 CF' ])
    df_raw = pd.DataFrame({'MODEL':model, 'F1 ERROR':f1e,'F2 ERROR':f2e, 'F3 ERROR':f3e})

    return df_res, df_raw


def change_label_set_1(datapoint_df, col):
    datapoint_df.at['E', col]= 'ɛ:' 
    datapoint_df.at['O', col]= 'ɔ:'
    datapoint_df.at['9', col]= 'œ:'
    datapoint_df.at['@', col]= 'ə:'
    datapoint_df.at['o', col]= 'o:'
    datapoint_df.at['a', col]= 'a:'
    datapoint_df.at['i', col]= 'i:'
    datapoint_df.at['e', col]= 'e:'
    datapoint_df.at['u', col]= 'u:'
    datapoint_df.at['A', col]= 'ɑ:'
    datapoint_df.at['2', col]= 'ø:'
    datapoint_df.at['U', col]= 'ʊ:'

    return datapoint_df

def custome_reindex_type2(df):
    return df.reindex(["a:", "i:", "u:","e:",'ɛ:','ə:','œ:','o:','ɔ:', 'ɑ:','ø:','ʊ:'])

def set_datapoint_index(df, col):
    df_temp = df.copy()
    df_temp['Label_idx'] = df_temp[col]
    return df_temp.set_index('Label_idx')

def compute_each_vowel_formant_eval_mono(exp_num):
    path = '../../experiment/result/eval_{}/formant/'.format(exp_num)

    a1 = np.load(join(path, 'actual_F1.npy'))
    a2 = np.load(join(path, 'actual_F2.npy'))
    a3 = np.load(join(path, 'actual_F3.npy'))

    e1 = np.load(join(path, 'estimated_F1.npy'))
    e2 = np.load(join(path, 'estimated_F2.npy'))
    e3 = np.load(join(path, 'estimated_F3.npy'))
    
    a1 = cut_head_and_tail(numpy_fillna(a1))
    a2 = cut_head_and_tail(numpy_fillna(a2))
    a3 = cut_head_and_tail(numpy_fillna(a3))

    e1 = cut_head_and_tail(numpy_fillna(e1))
    e2 = cut_head_and_tail(numpy_fillna(e2))
    e3 = cut_head_and_tail(numpy_fillna(e3))

    f1_error = np.mean(np.divide(np.absolute(a1 - e1), a1, out=np.zeros_like(a1), where=a1!=0.0)*100, axis=1)
    f2_error = np.mean(np.divide(np.absolute(a2 - e2), a2, out=np.zeros_like(a2), where=a2!=0.0)*100, axis=1)
    f3_error = np.mean(np.divide(np.absolute(a3 - e3), a3, out=np.zeros_like(a3), where=a3!=0.0)*100, axis=1)


    return pd.DataFrame({ 'F1 ERROR': f1_error,'F2 ERROR':f2_error, 'F3 ERROR':f3_error},columns=['F1 ERROR','F2 ERROR', 'F3 ERROR'])


def compute_each_vowel_formant_eval_di(exp_num):
    path = '../../experiment/result/eval_{}/formant/'.format(exp_num)


    with open('../../data/d_eval/syllable_name.txt') as f:
        label = np.array([word.strip()[0]+';'+word.strip()[1] for line in f for word in line.split(',')])

    a1 = np.load(join(path, 'actual_F1.npy'))
    a2 = np.load(join(path, 'actual_F2.npy'))
    a3 = np.load(join(path, 'actual_F3.npy'))

    e1 = np.load(join(path, 'estimated_F1.npy'))
    e2 = np.load(join(path, 'estimated_F2.npy'))
    e3 = np.load(join(path, 'estimated_F3.npy'))

    a1 = cut_head_and_tail(numpy_fillna(a1))
    a2 = cut_head_and_tail(numpy_fillna(a2))
    a3 = cut_head_and_tail(numpy_fillna(a3))

    e1 = cut_head_and_tail(numpy_fillna(e1))
    e2 = cut_head_and_tail(numpy_fillna(e2))
    e3 = cut_head_and_tail(numpy_fillna(e3))

    f1_error = np.divide(np.absolute(a1 - e1), a1, out=np.zeros_like(a1), where=a1!=0.0)*100
    f2_error = np.divide(np.absolute(a2 - e2), a2, out=np.zeros_like(a2), where=a2!=0.0)*100
    f3_error = np.divide(np.absolute(a3 - e3), a3, out=np.zeros_like(a3), where=a3!=0.0)*100


    f1_error = f1_error.reshape((f1_error.shape[0]//2,2,f1_error.shape[1]))
    f2_error = f2_error.reshape((f2_error.shape[0]//2,2,f2_error.shape[1]))
    f3_error = f3_error.reshape((f3_error.shape[0]//2,2,f3_error.shape[1]))

    f11_error = np.absolute(np.mean(f1_error[:,0,:], axis=1))
    f21_error = np.absolute(np.mean(f1_error[:,1,:], axis=1))
    f12_error = np.absolute(np.mean(f2_error[:,0,:], axis=1))
    f22_error = np.absolute(np.mean(f2_error[:,1,:], axis=1))
    f13_error = np.absolute(np.mean(f3_error[:,0,:], axis=1))
    f23_error = np.absolute(np.mean(f3_error[:,1,:], axis=1))


    res_df = pd.DataFrame({ 'Pho':label, '1F1 ERROR': f11_error,'1F2 ERROR':f12_error, '1F3 ERROR':f13_error,
                  '2F1 ERROR': f21_error,'2F2 ERROR':f22_error, '2F3 ERROR':f23_error})

    res_df['Pho1'], res_df['Pho2'] = res_df['Pho'].str.split(';', 1).str
    
    return res_df

def change_label_set_2(datapoint_df, col):
    datapoint_df.at['E', col]= 'ɛ:' 
    datapoint_df.at['O', col]= 'ɔ:'
    datapoint_df.at['7', col]= 'ɤ:'
    datapoint_df.at['M', col]= 'ɯ:'
    datapoint_df.at['o', col]= 'o:'
    datapoint_df.at['a', col]= 'a:'
    datapoint_df.at['i', col]= 'i:'
    datapoint_df.at['e', col]= 'e:'
    datapoint_df.at['u', col]= 'u:'
    return datapoint_df

def custome_reindex_type3(df):
    return df.reindex(["a:", "i:", "u:","e:",'ɛ:','ɯ:','ɤ:','o:','ɔ:'])

def set_datapoint_index(df, col):
    df_temp = df.copy()
    df_temp['Label_idx'] = df_temp[col]
    return df_temp.set_index('Label_idx')

def compute_each_vowel_formant_predict_di(exp_num):
    path = '../../experiment/result/predict_{}/formant/'.format(exp_num)


    with open('../../data/d_records/record_all/syllable_name.txt') as f:
        label = np.array([word.strip()[0]+';'+word.strip()[1] for line in f for word in line.split(',')])

    a1 = np.load(join(path, 'actual_F1.npy'))
    a2 = np.load(join(path, 'actual_F2.npy'))
    a3 = np.load(join(path, 'actual_F3.npy'))

    e1 = np.load(join(path, 'estimated_F1.npy'))
    e2 = np.load(join(path, 'estimated_F2.npy'))
    e3 = np.load(join(path, 'estimated_F3.npy'))

    a1 = cut_head_and_tail(numpy_fillna(a1)).astype('float')
    a2 = cut_head_and_tail(numpy_fillna(a2)).astype('float')
    a3 = cut_head_and_tail(numpy_fillna(a3)).astype('float')

    e1 = cut_head_and_tail(numpy_fillna(e1)).astype('float')
    e2 = cut_head_and_tail(numpy_fillna(e2)).astype('float')
    e3 = cut_head_and_tail(numpy_fillna(e3)).astype('float')

    f1_error = np.divide(np.absolute(a1 - e1), a1, out=np.zeros_like(a1), where=a1!=0.0)*100
    f2_error = np.divide(np.absolute(a2 - e2), a2, out=np.zeros_like(a2), where=a2!=0.0)*100
    f3_error = np.divide(np.absolute(a3 - e3), a3, out=np.zeros_like(a3), where=a3!=0.0)*100


    f1_error = f1_error.reshape((f1_error.shape[0]//2,2,f1_error.shape[1]))
    f2_error = f2_error.reshape((f2_error.shape[0]//2,2,f2_error.shape[1]))
    f3_error = f3_error.reshape((f3_error.shape[0]//2,2,f3_error.shape[1]))

    f11_error = np.absolute(np.mean(f1_error[:,0,:], axis=1))
    f21_error = np.absolute(np.mean(f1_error[:,1,:], axis=1))
    f12_error = np.absolute(np.mean(f2_error[:,0,:], axis=1))
    f22_error = np.absolute(np.mean(f2_error[:,1,:], axis=1))
    f13_error = np.absolute(np.mean(f3_error[:,0,:], axis=1))
    f23_error = np.absolute(np.mean(f3_error[:,1,:], axis=1))


    res_df = pd.DataFrame({ 'Pho':label, '1F1 ERROR': f11_error,'1F2 ERROR':f12_error, '1F3 ERROR':f13_error,
                  '2F1 ERROR': f21_error,'2F2 ERROR':f22_error, '2F3 ERROR':f23_error})

    res_df['Pho1'], res_df['Pho2'] = res_df['Pho'].str.split(';', 1).str
    
    return res_df


def get_min_max_mean_each_vowel_formant_predict_di(exp_num):
    path = '../../experiment/result/predict_{}/formant/'.format(exp_num)


    with open('../../data/d_records/record_all/syllable_name.txt') as f:
        label = np.array([word.strip()[0]+';'+word.strip()[1] for line in f for word in line.split(',')])

    a1 = np.load(join(path, 'actual_F1.npy'))
    a2 = np.load(join(path, 'actual_F2.npy'))

    e1 = np.load(join(path, 'estimated_F1.npy'))
    e2 = np.load(join(path, 'estimated_F2.npy'))

    a1 = cut_head_and_tail(numpy_fillna(a1)).astype('float')
    a2 = cut_head_and_tail(numpy_fillna(a2)).astype('float')

    e1 = cut_head_and_tail(numpy_fillna(e1)).astype('float')
    e2 = cut_head_and_tail(numpy_fillna(e2)).astype('float')
    
    a1 = a1.reshape((a1.shape[0]//2,2,a1.shape[1]))
    a2 = a2.reshape((a2.shape[0]//2,2,a2.shape[1]))
    
    a11_min = np.min(a1[:,0,:], axis=1)
    a12_min = np.min(a2[:,0,:], axis=1)
    a21_min = np.min(a1[:,1,:], axis=1)
    a22_min = np.min(a2[:,1,:], axis=1)

    a11_max = np.max(a1[:,0,:], axis=1)
    a12_max = np.max(a2[:,0,:], axis=1)
    a21_max = np.max(a1[:,1,:], axis=1)
    a22_max = np.max(a2[:,1,:], axis=1)
    
    a11_mean = np.mean(a1[:,0,:], axis=1)
    a12_mean = np.mean(a2[:,0,:], axis=1)
    a21_mean = np.mean(a1[:,1,:], axis=1)
    a22_mean = np.mean(a2[:,1,:], axis=1)
 


    res_df = pd.DataFrame({ 'Pho':label, 
                           '1F1 MIN': a11_min,'1F2 MIN':a12_min, '2F1 MIN': a21_min,'2F2 MIN':a22_min,
                           '1F1 MAX': a11_max,'1F2 MAX':a12_max, '2F1 MAX': a21_max,'2F2 MAX':a22_max,
                           '1F1 MEAN': a11_mean,'1F2 MEAN':a12_mean, '2F1 MEAN': a21_mean,'2F2 MEAN':a22_mean
                          })

    res_df['Pho1'], res_df['Pho2'] = res_df['Pho'].str.split(';', 1).str
    
    return res_df
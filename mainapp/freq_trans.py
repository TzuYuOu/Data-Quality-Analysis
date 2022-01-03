import ewtpy
import numpy as np
import pandas as pd
import os
import time

def bandpass_data(data, target_x, num_filter=5):
    data_bandpass_all = pd.DataFrame()
    num_filter = num_filter
    for target in target_x:
        ewt,  mfb ,boundaries = ewtpy.EWT1D(data[target], N=num_filter)
        data_bandpass = pd.DataFrame(ewt)
        data_bandpass.columns = ["%sbp_%s" %(target, i) for i in range(len(num_filter))]
        data_bandpass_all = pd.concat([data_bandpass_all, data_bandpass], axis=1)

    return data_bandpass_all

def Data_split(data, target_split):
    
    for target in target_split:
        # Seperate the data by sliding window
        data_split = {}
        for i in range(round(len(data)/window_size)):
            data_part = data[target][window_size*i:window_size*(i+1)]
            data_split.setdefault("split_%s" %i, data_part)
    
    return data_split

## 轉出時域特徵
def Features_time(data, window_size, target_trans):
    print("Feature of time domain transforming...")
    timestart = time.time()
    target_trans = target_trans
    feature_time_all = pd.DataFrame()
    
    column_name = ["mean", "max", "min", "std", "absmean", "kurt", "skew",
                   "waveindex", "pulseindex", "kurtindex", "peakindex", "sqra",
                   "marginindex", "skewindex", "crest", "clearance", "slope",
                   "shape", "impulse"]
    
    for target in target_trans:
        # Seperate the data by sliding window
        data_split = {}
        for i in range(round(len(data)/window_size)):
            data_part = data[target][window_size*i:window_size*(i+1)]
            data_split.setdefault("split_%s" %i, data_part)
        
        # Feature transformation
        feature_rms = []
        feature_mean = []
        feature_max = []
        feature_min = []
        feature_std = []
        feature_absmean = []
        feature_kurt = []
        feature_skew = []
        feature_waveindex = []
        feature_pulseindex = []
        feature_kurtindex = []
        feature_peakindex = []
        feature_sqra = []
        feature_marginindex = []
        feature_skewindex = []
        feature_crest = []
        feature_clearance = []
        feature_slope = []
        feature_shape = []
        feature_impulse = []
        feature_time_target = pd.DataFrame()
        
        for key in data_split.keys():
            # Statistic Feature
            avg = data_split[key].mean()
            maximum = data_split[key].max()
            minimum = data_split[key].min()
            std = data_split[key].std()
            absmean = abs(avg)
            kurt = data_split[key].kurt()
            skew = data_split[key].skew()
            rms = np.sqrt((data_split[key]**2).sum() / data_split[key].size)
            
            # Other Feature
            waveindex = rms / avg
            pulseindex = maximum / absmean
            kurtindex = kurt / rms
            peakindex = maximum / rms
            sqra = (np.sqrt(abs(data_split[key])).sum() / data_split[key].size)**2
            marginindex = maximum / sqra
            skewindex = skew / rms
            crest = maximum / rms
            clearance = maximum / np.sqrt((data_split[key]**0.5).sum() / data_split[key].size)
            shape = rms / avg
            impulse = maximum / avg
            x = [i for i in range(1, len(data_split[key])+1)]
            slope = np.polyfit(x, data_split[key], 1)[0]
            
            feature_mean.append(avg)
            feature_max.append(maximum)
            feature_min.append(minimum)
            feature_std.append(std)
            feature_absmean.append(absmean)
            feature_kurt.append(kurt)
            feature_skew.append(skew)
            feature_rms.append(rms)
            feature_waveindex.append(waveindex)
            feature_pulseindex.append(pulseindex)
            feature_kurtindex.append(kurtindex)
            feature_peakindex.append(peakindex)
            feature_sqra.append(sqra)
            feature_marginindex.append(marginindex)
            feature_skewindex.append(skewindex)
            feature_crest.append(crest)
            feature_clearance.append(clearance)
            feature_slope.append(slope)
            feature_shape.append(shape)
            feature_impulse.append(impulse)
            
        feature_rms = pd.DataFrame(feature_rms)
        feature_mean = pd.DataFrame(feature_mean)
        feature_max = pd.DataFrame(feature_max)
        feature_min = pd.DataFrame(feature_min)
        feature_std = pd.DataFrame(feature_std)
        feature_absmean = pd.DataFrame(feature_absmean)
        feature_kurt = pd.DataFrame(feature_kurt)
        feature_skew = pd.DataFrame(feature_skew)
        feature_waveindex = pd.DataFrame(feature_waveindex)
        feature_pulseindex = pd.DataFrame(feature_pulseindex)
        feature_kurtindex = pd.DataFrame(feature_kurtindex)
        feature_peakindex = pd.DataFrame(feature_peakindex)
        feature_sqra = pd.DataFrame(feature_sqra)
        feature_marginindex = pd.DataFrame(feature_marginindex)
        feature_skewindex = pd.DataFrame(feature_skewindex)
        feature_crest = pd.DataFrame(feature_crest)
        feature_clearance = pd.DataFrame(feature_clearance)
        feature_slope = pd.DataFrame(feature_slope)
        feature_shape = pd.DataFrame(feature_shape)
        feature_impulse = pd.DataFrame(feature_impulse)
            
        feature_time_target = pd.concat([feature_time_target, 
                                         feature_mean, 
                                         feature_max, 
                                         feature_min,
                                         feature_std,
                                         feature_absmean,                                      
                                         feature_kurt,
                                         feature_skew,
                                         feature_waveindex,
                                         feature_pulseindex,
                                         feature_kurtindex,
                                         feature_peakindex,
                                         feature_sqra,
                                         feature_marginindex,
                                         feature_skewindex,
                                         feature_crest,
                                         feature_clearance,
                                         feature_slope,
                                         feature_shape,
                                         feature_impulse
                                         ], 
                                        axis=1)
        
        feature_time_target.columns = ["%s_" %target + name for name in column_name]
        
        feature_time_all = pd.concat([feature_time_all, feature_time_target], axis=1)
    
    timefinish = time.time()
    print("Transfomation accomplish. Using %f s." %(timefinish-timestart))
    
    return feature_time_all

# 轉出頻域特徵
def Feature_freq(data, window_size, target_trans, sampling_freq):
    print("Feature of frequence domain transforming...")
    timestart = time.time()
    feature_freq_all = pd.DataFrame()
    
    column_name = ["_F_amp1", "_F_amp1_f", "_F_amp2", "_F_amp2_f", "_F_amp_mean", "_F_amp_var", "_F_amp_skew", "_F_amp_kurt"]
    
    for target in target_trans:
        # Seperate the data by sliding window
        data_split = {}
        for i in range(round(len(data)/window_size)):
            data_part = data[target][window_size*i:window_size*(i+1)]
            data_split.setdefault("split_%s" %i, data_part)
    
        feature_freq_target = pd.DataFrame()
        for key in data_split.keys():
            # 振福
            fft_amp = np.fft.fft(data_split[key])[:round(data_split[key].size / 2)]
            # 頻率
            fft_freq = np.fft.fftfreq(data_split[key].size, 1/sampling_freq)[:round(data_split[key].size / 2)]
            
            df_fft = pd.DataFrame(fft_freq, abs(fft_amp)).reset_index().sort_values(by="index",ascending=False)
            df_fft.columns = ["magn", "frequency"]
            df_fft = df_fft[df_fft['frequency']>0]
            df_fft = df_fft.reset_index(drop=True)
            first = df_fft.iloc[0]
            amp1,amp1_f = first[0],first[1]
            second = df_fft.iloc[1]
            amp2,amp2_f = second[0],second[1]
            amp_mean = df_fft['magn'].mean()
            amp_var = df_fft['magn'].std()
            amp_skew = df_fft['magn'].skew()
            amp_kurt = df_fft['magn'].kurt()
            list_col = [amp1,amp1_f,amp2,amp2_f,amp_mean,amp_var,amp_skew,amp_kurt]
    
            df_col = pd.DataFrame(list_col).T
            feature_freq_target = pd.concat([feature_freq_target, df_col])
            
        feature_freq_target.columns = ["%s" %target + name for name in column_name]
            
        feature_freq_all = pd.concat([feature_freq_all, feature_freq_target], axis=1)
    feature_freq_all = feature_freq_all.reset_index(drop=True)
    
    timefinish = time.time()
    print("Transfomation accomplish. Using %f s." %(timefinish-timestart))
    
    return feature_freq_all      

## 轉出所有特徵
def Feature_all(data, window_size, target_trans, sampling_freq):    

    time_all_correct = Features_time(data, window_size, target_trans)
    
    fft_spt = Feature_freq(data, window_size, target_trans, sampling_freq)
    
    data_allfeature = pd.concat([time_all_correct,fft_spt],axis=1)
    
    return data_allfeature

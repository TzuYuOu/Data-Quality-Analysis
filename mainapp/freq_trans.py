
import ewtpy
import numpy as np
import pandas as pd
import os
import time


def bandpass_data(data, target_x, target_y, num_filter=5):
    data_bandpass_all = pd.DataFrame()
    num_filter = num_filter
    for target in target_x:
        ewt,  mfb, boundaries = ewtpy.EWT1D(data[target], N=num_filter)
        data_bandpass = pd.DataFrame(ewt)
        data_bandpass.columns = ["%sbp_%s" % (target, i) for i in range(num_filter)]
        data_bandpass_all = pd.concat([data_bandpass_all, data_bandpass], axis=1)
    data_bandpass_all = pd.concat([data_bandpass_all, data[target_y]], axis=1)
    return data_bandpass_all


def Data_split(data, target_split, target_y, window_size):

    data_split = {}
    for i in range(round(len(data)/window_size)):
        target_trans = target_split + target_y
        data_part = data[target_trans][window_size*i:window_size*(i+1)]
        data_split.setdefault("split_%s" % i, data_part)

    return data_split


# 轉出時域特徵
def Features_time(data, target_trans, target_y):
    print("Feature of time domain transforming...")
    timestart = time.time()
    target_trans = target_trans

    column_name = ["mean", "max", "min", "std", "absmean", "kurt", "skew",
                   "waveindex", "pulseindex", "kurtindex", "peakindex", "sqra",
                   "marginindex", "skewindex", "crest", "clearance", "slope",
                   "shape", "impulse"]
    target_y = target_y[0]
    data_split = data

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

    feature_y = []

    for key in data_split.keys():
        # Statistic Feature
        avg = pd.Series(data_split[key][target_trans].mean())
        maximum = pd.Series(data_split[key][target_trans].max())
        minimum = pd.Series(data_split[key][target_trans].min())
        std = pd.Series(data_split[key][target_trans].std())

        kurt = pd.Series(data_split[key][target_trans].kurt())
        skew = pd.Series(data_split[key][target_trans].skew())
        rms = pd.Series(np.sqrt((data_split[key][target_trans]**2).sum() / data_split[key][target_trans].size))
        absmean = abs(avg)

        # Other Feature
        waveindex = rms / avg
        pulseindex = maximum / absmean
        kurtindex = kurt / rms
        peakindex = maximum / rms
        sqra = (np.sqrt(abs(data_split[key][target_trans])).sum() / data_split[key][target_trans].size)**2
        marginindex = maximum / sqra
        skewindex = skew / rms
        crest = maximum / rms
        clearance = maximum / np.sqrt((data_split[key][target_trans]**0.5).sum() / data_split[key][target_trans].size)
        shape = rms / avg
        impulse = maximum / avg
        x = [i for i in range(1, len(data_split[key][target_trans])+1)]
        slope = np.polyfit(x, data_split[key][target_trans], 1)[0]

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
        feature_y.append(float(data_split[key][target_y][-1:]))

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
    feature_y = pd.DataFrame(feature_y)

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
                                     feature_impulse,
                                     feature_y
                                     ],
                                    axis=1)
    feature_name = []
    for name in column_name:
        feature_name.extend(["%s_" % target + name for target in target_trans])
    feature_time_target.columns = feature_name + [target_y]

    timefinish = time.time()
    print("Transfomation accomplish. Using %f s." % (timefinish-timestart))

    return feature_time_target


# 轉出頻域特徵
def Features_freq(data, window_size, target_trans, target_y, sampling_freq):
    print("Feature of frequence domain transforming...")
    timestart = time.time()
    feature_freq_all = pd.DataFrame()

    column_name = ["_F_amp1",
                   "_F_amp1_f",
                   "_F_amp2",
                   "_F_amp2_f",
                   "_F_amp_mean",
                   "_F_amp_var",
                   "_F_amp_skew",
                   "_F_amp_kurt"]

    target_y = target_y[0]
    data_split = data
    y = []
    for key in data_split.keys():
        feature_freq_trans = pd.DataFrame()
        for target in target_trans:
            feature_freq_target = pd.DataFrame()
            # 振福
            fft_amp = np.fft.fft(data_split[key][target])[:round(data_split[key][target].size / 2)]
            # 頻率
            fft_freq = np.fft.fftfreq(data_split[key][target].size,
                                      1/sampling_freq)[:round(data_split[key][target].size / 2)]

            df_fft = pd.DataFrame(fft_freq, abs(fft_amp)).reset_index().sort_values(by="index",
                                                                                    ascending=False)
            df_fft.columns = ["magn", "frequency"]
            df_fft = df_fft[df_fft['frequency'] > 0]
            df_fft = df_fft.reset_index(drop=True)
            first = df_fft.iloc[0]
            amp1, amp1_f = first[0], first[1]
            second = df_fft.iloc[1]
            amp2, amp2_f = second[0], second[1]
            amp_mean = df_fft['magn'].mean()
            amp_var = df_fft['magn'].std()
            amp_skew = df_fft['magn'].skew()
            amp_kurt = df_fft['magn'].kurt()

            list_col = [amp1, amp1_f, amp2, amp2_f,
                        amp_mean, amp_var, amp_skew, amp_kurt]

            df_col = pd.DataFrame(list_col).T
            feature_freq_target = pd.concat([feature_freq_target, df_col])
            feature_freq_target.columns = ["%s" % target + name for name in column_name]
            feature_freq_trans = pd.concat([feature_freq_trans, feature_freq_target], axis=1)

        feature_freq_all = pd.concat([feature_freq_all, feature_freq_trans], axis=0)
        y.append(float(data_split[key][target_y][-1:]))
    feature_freq_all[target_y] = y
    feature_freq_all = feature_freq_all.reset_index(drop=True)

    timefinish = time.time()
    print("Transfomation accomplish. Using %f s." % (timefinish-timestart))

    return feature_freq_all


# 轉出所有特徵
def Feature_all(data, window_size, target_trans, target_y, sampling_freq):

    data = Data_split(data, target_trans, target_y, window_size)

    time_all_correct = Features_time(data, target_trans, target_y)

    fft_spt = Features_freq(data, window_size, target_trans, target_y, sampling_freq)
    fft_spt = fft_spt.drop(target_y, axis=1)

    data_allfeature = pd.concat([time_all_correct, fft_spt], axis=1)

    return data_allfeature


# if __name__ == "__main__":

#     path = "C:\\Users\\Kevin\\Desktop\\Code Project\\Python\\DQA"
#     filelist = os.listdir(path)
#     datapath = os.path.join(path, filelist[1])
#     df = pd.read_csv(datapath)

#     df_seperate = Data_split(df, ["vibrationsd_5d", "rotatesd_5d"], ["RUL_I"], window_size=100)
#     df_timefea = Features_time(df_seperate, ["vibrationsd_5d", "rotatesd_5d"], ["RUL_I"])
#     df_feqfea = Features_freq(df_seperate,
#                               100,
#                               ["vibrationsd_5d", "rotatesd_5d"],
#                               ["RUL_I"],
#                               100)
#     df_band = bandpass_data(df, ["vibrationsd_5d", "rotatesd_5d"], ["RUL_I"])

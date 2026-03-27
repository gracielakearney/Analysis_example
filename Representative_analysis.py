# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:41:46 2026

@author: Grace
"""

import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from collections import OrderedDict
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy.stats import zscore


#%%

def cut_file(arr_dict, time, interval):
    time_mask = (time > interval[0]) & (time < interval[1])
    selected_time = time[time_mask]
    selected_arr_dict = {key:arr_dict[key][time_mask] for key in arr_dict.keys()}
    return selected_arr_dict, selected_time

def units_in_dp(channel_dict):
    if len(np.unique(list(channel_dict.values()))) == 1:
        unit_dict = {0: list(channel_dict.keys())}
    else:
        unit_dict = {ganglion:[] for ganglion in [1,2,3]}
        for key in channel_dict.keys():
            for ganglion in [1,2,3]:
                if channel_dict[key] == ganglion-1:
                    unit_dict[ganglion].append(key)
    return unit_dict

def bin_spikes(cleaned_dict, time, step):
    bins = np.arange(time[0], time[-1] + step, step).round(2)
    #np.digitize devuelve el índice del bin en el que se encuentra cada elemento de un array
    binned_spikes = {}
    for unit, idx_arr in cleaned_dict.items():
        bin_idxs = np.digitize(time[idx_arr], bins)
        count = dict(Counter(bin_idxs)) #cuento cuántos spikes hay en cada bin, {bin:#spikes}
        #agrego los bines en los que no hay spikes
        idx_list = list(np.arange(0, len(bins)))   
        for i in idx_list:
            if i not in count.keys():
                count[i] = 0
        #ordeno las claves del diccionario
        count = dict(OrderedDict(sorted(count.items())))
        #asocio los #spikes a los bines
        values = np.array([value for value in count.values()])
        binned_spikes[int(unit)] = np.vstack((bins, values))
    return binned_spikes    
    
def spike_frequency(cleaned_dict, time, step):
    binned_spikes = bin_spikes(cleaned_dict, time, step)
    spike_freq = {}
    for unit in binned_spikes.keys():
        spike_freq[int(unit)] = np.vstack((binned_spikes[unit][0], binned_spikes[unit][1] / step))
    return spike_freq

def slice_frequency(spike_freq, limit_list):
    sliced_freq = []
    for limits in limit_list:
        start = limits[0]
        end = limits[1]
        interval_mask = (spike_freq[1][0] >= start) & (spike_freq[1][0] < end)
        sliced_freq.append({unit:spike_freq[:, interval_mask] for unit, spike_freq
                           in spike_freq.items()})
       
    return sliced_freq

def filter_freq(spike_freq, sigma):
    '''
    Aplica el filtro gaussiano de scipy.ndimage. Lo estrictamente correcto
    hubiera sido contruir un kernel con signal.windows.gaussian y luego
    filtrar la señal con signal.convolve. Usando sigma = 3 con esta función
    es equivalente a usar un kernel gaussiano normalizado de N=19 y sigma=3
    (el código está comentado debajo de esta función)

    Parameters
    ----------
    spike_freq : dict
        Diccionario con unidad:frecuencia de spikes.
    sigma : scalar
        Standard deviation for Gaussian kernel.

    Returns
    -------
    filtered_freq : dict
        Diccionario con unidad:frecuencia de spikes filtrada.

    '''
    filtered_freq = {unit:np.vstack((spike_freq[unit][0], gaussian_filter(spike_freq[unit][1], sigma)))
                     for unit in spike_freq.keys()}
    return filtered_freq

def plot_spikes(arr_dict, time, spike_idx_dict, color_dict, unit_dict, save=None):
    fig, axs = plt.subplots(figsize=(11.8, 5.9),nrows=len(arr_dict), sharex=True)
    plt.subplots_adjust(top=0.976,bottom=0.063,left=0.061,right=0.908,hspace=0.148)
    if len(arr_dict) == 1:
        axs.plot(time, arr_dict[0], lw=0.7, c='k')
        for unit in unit_dict[0]:
            axs.scatter(time[spike_idx_dict[unit]], 
                        arr_dict[0][spike_idx_dict[unit]],
                        s=30, color=color_dict[unit],
                        label=unit)
        axs.legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    else:
        for i, ganglion in enumerate(arr_dict.keys()):
            axs[i].plot(time, arr_dict[ganglion], lw=0.7, c='k')
            for unit in unit_dict[ganglion]:
                axs[i].scatter(time[spike_idx_dict[unit]], 
                               arr_dict[ganglion][spike_idx_dict[unit]],
                               s=30, color=color_dict[unit],
                               label=unit)
            axs[i].legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save)
    
def plot_spike_freq(filtered_freq, color_dict, save=None):
    fig, axs = plt.subplots(figsize=(11.8, 5.9), nrows=len(filtered_freq), sharex=True)
    for i, unit in enumerate(filtered_freq.keys()):
        axs[i].plot(filtered_freq[unit][0], filtered_freq[unit][1], color=color_dict[unit], lw=0.7, label=unit)
        axs[i].legend(bbox_to_anchor=(1.01, 1.0), loc='upper left')
    plt.tight_layout()
    
    if save is not None:
        plt.savefig(save)


def cut_and_resize(selected_freq, cut_point_list):
    
    sliced_freq = []
    for i in range(len(cut_point_list)-1):
        sliced_freq.append({unit:spike_freq[:, cut_point_list[i]:cut_point_list[i+1]]
                            for unit, spike_freq in selected_freq.items()})
        
    max_length = max([len(spike_freq_dict[0][0]) for spike_freq_dict in sliced_freq])
    
    resized_freq = []
    for spike_freq_dict in sliced_freq:
        resized_freq.append({unit:resize_proportional(freq[1], max_length)
                             for unit, freq in spike_freq_dict.items()}) 
        
    return resized_freq

def match_extrema(min_list, max_list):
        
    matched_list = []
    for i in range(len(max_list)):
        matched_dict = {}
        left_mins = []
        for min_idx in min_list:
            if min_idx < max_list[i]:
                left_mins.append(min_idx)
                
        right_mins = []
        for min_idx in min_list:
            if min_idx > max_list[i]:
                right_mins.append(min_idx)  
        
        try:
            matched_dict['left min idx'] = left_mins[-1]
        except IndexError:
            print("Left minimum is missing, first peak won't be considered")
            continue
        
        matched_dict['max idx'] = max_list[i]
        
        try:
            matched_dict['right min idx'] = right_mins[0]
            matched_list.append(matched_dict)
        except IndexError:
            print("Right minimum is missing, last peak won't be considered")    
    
    return matched_list

    
def peak_prominence(unit_selected_freq, min_list, max_list):
    
    matched_extrema = match_extrema(min_list, max_list)
    peak_list = [{'left min' : unit_selected_freq[idx_dict['left min idx']],
                  'max' : unit_selected_freq[idx_dict['max idx']],
                  'prominence' : (unit_selected_freq[idx_dict['max idx']] - 
                                  unit_selected_freq[idx_dict['left min idx']])}
                 for idx_dict in matched_extrema]
        
    df_prominence = pd.DataFrame(peak_list)
    
    return df_prominence
        
def make_windows(window_size, step, limit_list):
    window_list = []
    for i in range(0, len(limit_list) - window_size, step): 
        window_list.append((limit_list[i], limit_list[i + window_size]))
    
    return window_list        


def zscore_fragments(selected_freq, window_list, unit_list, ref_unit):
    zscore_fragment_list = []
    for window in window_list:
        window_dict = {}
        window_dict['time'] = (selected_freq[ref_unit][0][window[0]] + 
                              (selected_freq[ref_unit][0][window[1]] - selected_freq[ref_unit][0][window[0]]) / 2)
        for unit in unit_list:   
            window_dict[unit] = zscore(selected_freq[unit][1][window[0]:window[1]])
            
        zscore_fragment_list.append(window_dict)
    
    return zscore_fragment_list

def cross_correlation(zscore_fragment_list, correlation_pairs, bin_fs):
    xcorr_list = []
    for fragment in zscore_fragment_list:
        fragment_dict = {}
        fragment_dict['time'] = fragment['time']
        for pair in correlation_pairs:
            label = ' ' + str(pair[0]) + str(pair[1])
            if not np.any(np.isnan(fragment[pair[0]])) | np.any(np.isnan(fragment[pair[1]])):
                fragment_dict['corr' + label] = correlate(fragment[pair[0]], fragment[pair[1]], mode='same') / len(fragment[pair[0]])
                fragment_dict['lags' + label] = correlation_lags(len(fragment[pair[0]]), len(fragment[pair[1]]), mode='same') / bin_fs
            else:
                fragment_dict['corr' + label] = np.nan
                fragment_dict['lags' + label] = np.nan
        xcorr_list.append(fragment_dict)
    
    return xcorr_list

def calculate_99_percent_points(selected_spike_freq_arr, min_arr, max_arr):
    
    matched_extrema = match_extrema(min_arr, max_arr)
        
    ninety_nine_percent_values = np.round([(selected_spike_freq_arr[idx_dict['max idx']] - selected_spike_freq_arr[idx_dict['right min idx']]) * 0.01
                                           + selected_spike_freq_arr[idx_dict['right min idx']] for idx_dict in matched_extrema], 2)
    
    ninety_nine_signal_distance = [np.abs(selected_spike_freq_arr[idx_dict['max idx']:idx_dict['right min idx']] - ninety_nine_percent_values[i])
                                   for i, idx_dict in enumerate(matched_extrema)]
    
    ninety_nine_percent_points = [np.argmin(ninety_nine_signal_distance[i]) + matched_extrema[i]['max idx']
                                  for i in range(len(ninety_nine_percent_values))]      
    
    return ninety_nine_percent_points

def calculate_left_half_width(selected_spike_freq_arr, min_arr, max_arr):
    
    matched_extrema = match_extrema(min_arr, max_arr)
        
    fifty_percent_values = np.round([(selected_spike_freq_arr[idx_dict['max idx']] - selected_spike_freq_arr[idx_dict['left min idx']]) * 0.5
                                    + selected_spike_freq_arr[idx_dict['left min idx']] for idx_dict in matched_extrema], 2)

    #como la frecuencia de spikes puede no tener el valor idéntico al 1% calculado, encuentro el
    # valor mínimo de la diferencia etre el 50% calculado y la señal
    fifty_perc_signal_distance = [np.abs(selected_spike_freq_arr[idx_dict['left min idx']:idx_dict['max idx']] - fifty_percent_values[i])
                                  for i, idx_dict in enumerate(matched_extrema)]
    
    half_width_points = [np.argmin(fifty_perc_signal_distance[i]) + matched_extrema[i]['left min idx']
                         for i in range(len(fifty_percent_values))]
        
    return half_width_points

def calculate_right_half_width(selected_spike_freq_arr, min_arr, max_arr):
    
    matched_extrema = match_extrema(min_arr, max_arr)
        
    fifty_percent_values = np.round([(selected_spike_freq_arr[idx_dict['max idx']] - selected_spike_freq_arr[idx_dict['right min idx']]) * 0.5
                                     + selected_spike_freq_arr[idx_dict['right min idx']] for idx_dict in matched_extrema], 2)
    
    fifty_perc_signal_distance = [np.abs(selected_spike_freq_arr[idx_dict['max idx']:idx_dict['right min idx']] - fifty_percent_values[i])
                                  for i, idx_dict in enumerate(matched_extrema)]
    
    half_width_points = [np.argmin(fifty_perc_signal_distance[i]) + matched_extrema[i]['max idx']
                         for i in range(len(fifty_percent_values))]      
    
    return half_width_points

def half_width(selected_spike_freq, min_arr, max_arr): #arr with shape[0] = 2
    
    left_half_widths = calculate_left_half_width(selected_spike_freq[1], min_arr, max_arr)
    right_half_widths = calculate_right_half_width(selected_spike_freq[1], min_arr, max_arr)
    
    left_hw_times = selected_spike_freq[0][left_half_widths]
    right_hw_times = selected_spike_freq[0][right_half_widths]
    
    half_widths = right_hw_times - left_hw_times
    
    return half_widths

def resize_proportional(arr, normalized_max, new_length):
    return np.interp(np.linspace(0, normalized_max, new_length), np.linspace(0, normalized_max, len(arr)), arr)


def resize_sliced_freq(sliced_freq, normalized_max, ref_unit): #en cada intervalo ref_unit siempre está
         
    max_length = max([len(spike_freq_dict[ref_unit][0]) for spike_freq_dict in sliced_freq])
    
    resized_freq = []
    for spike_freq_dict in sliced_freq:
        resized_freq.append({unit:resize_proportional(freq[1], normalized_max, max_length)
                              for unit, freq in spike_freq_dict.items()}) 
        
    return resized_freq


def normalize_cycles(selected_spike_freq, end_point_list, ref_unit):
    
    #inicio y fin de cada ciclo en índices
    cycle_limits = make_windows(1, 1, end_point_list) #cada ciclo corresponde a una ventana de size=1, step=1 delimitados por end_point_list
    
    #inicio y fin de cada ciclo en segundos
    cycle_limits_in_s = [(selected_spike_freq[1][0][cycle[0]], selected_spike_freq[1][0][cycle[1]])
                          for cycle in cycle_limits]
    
    #corto cada ciclo
    sliced_cycles = slice_frequency(selected_spike_freq, cycle_limits_in_s)
    
    #normalizo cada ciclo de 0 a 1 con el número de puntos del ciclo más largo
    norm_cycles = resize_sliced_freq(sliced_cycles, 1, ref_unit) #normalized_max=1
    
    return norm_cycles  

def paste_cycles(normalized_cycles, interval_length):
    pasted_cycles = []
    for i in range(len(normalized_cycles) - (interval_length - 1)):
        norm_dict = {}
        for unit in normalized_cycles[0].keys():
            norm_dict[unit] = np.concatenate([normalized_cycles[j][unit] for j in range(i, i+3, 1)])
        pasted_cycles.append(norm_dict)
    
    return pasted_cycles

def max_peak_in_section(lag_arr, xcorr_list, max_lag, min_xcorr_value):
    lag_mask = abs(lag_arr) <= max_lag
    
    #selecciono dónde buscar los picos
    selected_xcorr = [xcorr_arr[lag_mask] for xcorr_arr in xcorr_list] 
    
    #encuentro los índices de los máximos locales y me quedo con el del pico más grande
    max_xcorr_idxs = []
    for xcorr_arr in selected_xcorr:
        local_max_idxs, max_dict = find_peaks(xcorr_arr, height=min_xcorr_value) #puede dar varios idx
        if len(local_max_idxs) > 0:
            max_idx = local_max_idxs[np.argmax(max_dict['peak_heights'])] #me quedo con el pico más grande
        else:
            max_idx = np.nan
        max_xcorr_idxs.append(max_idx) 
    
    #computo los valores de xcorr y lag para cada intervalo
    max_xcorr_peaks = []
    for i, idx in enumerate(max_xcorr_idxs):
        max_xcorr_dict = {}
        if not np.isnan(idx):
            max_xcorr_dict['xcorr'] = selected_xcorr[i][idx]
            max_xcorr_dict['lag'] = lag_arr[lag_mask][idx]
        else:
            max_xcorr_dict['xcorr'] = np.nan
            max_xcorr_dict['lag'] = np.nan
        max_xcorr_peaks.append(max_xcorr_dict)
    
    max_xcorr_peaks = pd.DataFrame(max_xcorr_peaks)
    
    return max_xcorr_peaks

def min_peak_in_section(lag_arr, xcorr_list, max_lag, min_xcorr_value):
    lag_mask = abs(lag_arr) <= max_lag
    
    #selecciono dónde buscar los picos
    selected_xcorr = [xcorr_arr[lag_mask] for xcorr_arr in xcorr_list] 
    
    #encuentro los índices de los máximos locales y me quedo con el del pico más grande
    min_xcorr_idxs = []
    for xcorr_arr in selected_xcorr:
        local_min_idxs, min_dict = find_peaks(-xcorr_arr, height=min_xcorr_value) #puede dar varios idx
        if len(local_min_idxs) > 0:
            min_idx = local_min_idxs[np.argmin(-min_dict['peak_heights'])] #me quedo con el pico más chico
        else:
            min_idx = np.nan
        min_xcorr_idxs.append(min_idx) 
    
    #computo los valores de xcorr y lag para cada intervalo
    min_xcorr_peaks = []
    for i, idx in enumerate(min_xcorr_idxs):
        min_xcorr_dict = {}
        if not np.isnan(idx):
            min_xcorr_dict['xcorr'] = selected_xcorr[i][idx]
            min_xcorr_dict['lag'] = lag_arr[lag_mask][idx]
        else:
            min_xcorr_dict['xcorr'] = np.nan
            min_xcorr_dict['lag'] = np.nan
        min_xcorr_peaks.append(min_xcorr_dict)
    
    min_xcorr_peaks = pd.DataFrame(min_xcorr_peaks)
    
    return min_xcorr_peaks

def mean_resultant_length(angles):
    C = np.nanmean(np.cos(angles))
    S = np.nanmean(np.sin(angles))
    return np.hypot(C, S)  # R ∈ [0,1]

def local_de3_parameters(stable_spike_freq_dict, cycle_limits, stable_min_dict, stable_max_dict, local_unit):
    
    exp_props = {}
    #hay que hacer el rango hasta +3 porque las ventanas de xcorr tienen un tamaño de 3 ciclos 
    exp_props['period'] = [stable_spike_freq_dict[local_unit][0][cycle_limits[i+1]] - 
                                              stable_spike_freq_dict[local_unit][0][cycle_limits[i]]
                                              for i in range(len(cycle_limits)-1)]

    #lo calcula desde el primer burst (el que queda antes del primer límite),
    #así que hay que empezar desde window_start_idx_22070801_v3 + 1
    exp_props['hw'] = half_width(stable_spike_freq_dict[local_unit], 
                                                    stable_min_dict[local_unit],
                                                    stable_max_dict[local_unit])

    exp_props['dc'] = exp_props['hw'] / exp_props['period']

    exp_props['sf'] = peak_prominence(stable_spike_freq_dict[local_unit][1],
                                                         stable_min_dict[local_unit],
                                                         stable_max_dict[local_unit]).loc[:, 'prominence'].values
    
    exp_props = pd.DataFrame(exp_props)
    exp_props.insert(loc=0, column='unit', value=local_unit)
    exp_props['unit'] = exp_props['unit'].astype('category')
    
    return exp_props

def spike_freq_amp_in_chain(stable_spike_freq_dict, cycle_limits, stable_min_dict, stable_max_dict, unit_list):
    
    last_idx = cycle_limits[-1] #último valor dentro de la región estable
    
    spike_freq_amp = {}
    
    for unit in unit_list:
        unit_maxs = [idx for idx in stable_max_dict[unit] if idx <= last_idx] #que ningún pico quede luego del final de la región estable
    
        spike_freq_amp[f'{unit}'] = peak_prominence(stable_spike_freq_dict[unit][1],
                                                    stable_min_dict[unit],
                                                    unit_maxs).loc[:, 'prominence'].values
    
    spike_freq_amp_df = pd.Series(spike_freq_amp).explode().reset_index().set_axis(['unit', 'spike_freq_amp'], axis=1)
    
    spike_freq_amp_df = spike_freq_amp_df.astype({"unit": 'category', "spike_freq_amp": float})
    
    return spike_freq_amp_df

#%%
#variables de uso global

# Definir el colormap personalizado: de verde (mínimo) a rojo (máximo)
custom_cmap = LinearSegmentedColormap.from_list("yellow_to_violet", ["yellow", "mediumslateblue"])

unit_names = {1:'DE3 1', 6:'DE3 2', 11:'DE3 3', 2:'anti 1', 7:'anti 2', 12:'anti 3'}

xcorr_colors = {(1,6): 'r',
                (6,11): 'purple',
                (1,11): 'darkgoldenrod'}

xcorr_pair_names = {(1,6): 'DE3 1 - DE3 2',
                    (6,11): 'DE3 2 - DE3 3',
                    (1,11): 'DE3 1 - DE3 3'}
#%%
    
with open('arr_dict_25120801.pickle', 'rb') as handle:
    arr_dict_25120801 = pickle.load(handle)
    
with open('time_25120801.pickle', 'rb') as handle:
    time_25120801 = pickle.load(handle)

fs = 20200
    
arr_dict_25120801, time_25120801 = cut_file(arr_dict_25120801, time_25120801, [130, 790])
arr_dict_25120801 = dict(zip([1,2,3], arr_dict_25120801.values()))


color_dict_25120801 = {1:'b',
                        6:'darkorange',
                        11:'g',
                        2: 'r',
                        7:'purple',
                        12:'saddlebrown',
                        3: 'grey',
                        8: 'indianred',
                        13:'mediumpurple',
                        4:'hotpink',
                        9: 'olive',
                        14:'crimson',
                        5:'lightseagreen',
                        10:'dodgerblue',
                        15:'darkkhaki',
                        16:'violet',
                        17:'lawngreen'}

# Construyendo una ventana gaussiana
sigma_tiempo = 0.0002   # el sigma definido en segundos (buscamos un valor que suavize el ruido sin deformar la señal)
n_sigmas = 3  # cuántos sigmas incluir de cada lado
dt=1/fs
sigma = int(sigma_tiempo/dt)   # sigma (en cantidad de puntos)
N = 2*n_sigmas*sigma+1   # siempre impar!
kernel = signal.windows.gaussian(N, sigma)
kernel = kernel/sum(kernel)

# Suavizando
filt_g1 = signal.convolve(arr_dict_25120801[1], kernel, mode='same')  
filt_g2 = signal.convolve(arr_dict_25120801[2], kernel, mode='same')  
filt_g3 = signal.convolve(arr_dict_25120801[3], kernel, mode='same')   
  

unit1, unit1_props = find_peaks(-filt_g1, height=(0.38, 0.55), prominence=(0.75, 0.95),
                                rel_height=0.3, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit1_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit1_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit1_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit1_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit1_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit1_props), x='widths', bins=50, ax=axs[1,2])

unit2, unit2_props = find_peaks(filt_g1, height=(0.07, 0.12), prominence=(0.2, 0.25),
                                rel_height=0.25, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit2_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit2_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit2_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit2_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit2_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit2_props), x='widths', bins=50, ax=axs[1,2])

unit3, unit3_props = find_peaks(-filt_g1, height=(0.3, 0.38), prominence=(0.6, 0.72),
                                rel_height=0.3, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit3_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit3_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit3_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit3_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit3_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit3_props), x='widths', bins=50, ax=axs[1,2])

unit4, unit4_props = find_peaks(filt_g1, height=(0.04, 0.08), prominence=(0.09, 0.17),
                                rel_height=0.25, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit4_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit4_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit4_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit4_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit4_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit4_props), x='widths', bins=50, ax=axs[1,2])

unit5, unit5_props = find_peaks(filt_g1, height=(0.2, 0.26), prominence=(0.44, 0.53),
                                rel_height=0.3, width=2, distance=25)


# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit5_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit5_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit5_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit5_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit5_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit5_props), x='widths', bins=50, ax=axs[1,2])

# fig, ax = plt.subplots()
# ax.plot(time_25120801, filt_g1, c='k', lw=0.6)
# ax.scatter(time_25120801[unit1], filt_g1[unit1], color=color_dict_25120801[1])
# ax.scatter(time_25120801[unit2], filt_g1[unit2], color=color_dict_25120801[2])
# ax.scatter(time_25120801[unit3], filt_g1[unit3], color=color_dict_25120801[3])
# ax.scatter(time_25120801[unit4], filt_g1[unit4], color=color_dict_25120801[4])
# ax.scatter(time_25120801[unit5], filt_g1[unit5], color=color_dict_25120801[5])

unit6, unit6_props = find_peaks(filt_g2, height=(0.18, 0.34), prominence=(0.55, 0.7),
                                rel_height=0.3, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit6_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit6_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit6_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit6_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit6_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit6_props), x='widths', bins=50, ax=axs[1,2])

unit7, unit7_props = find_peaks(filt_g2, height=(0.08, 0.12), prominence=(0.18, 0.26),
                                rel_height=0.3, width=25, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit7_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit7_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit7_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit7_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit7_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit7_props), x='widths', bins=50, ax=axs[1,2])

unit8, unit8_props = find_peaks(filt_g2, height=(0.1, 0.2), prominence=(0.28, 0.4),
                                rel_height=0.3, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit8_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit8_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit8_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit8_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit8_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit8_props), x='widths', bins=50, ax=axs[1,2])


# fig, ax = plt.subplots()
# ax.plot(time_25120801, filt_g2, c='k', lw=0.6)
# ax.scatter(time_25120801[unit6], filt_g2[unit6], color=color_dict_25120801[6])
# ax.scatter(time_25120801[unit7], filt_g2[unit7], color=color_dict_25120801[7])
# ax.scatter(time_25120801[unit8], filt_g2[unit8], color=color_dict_25120801[8])

unit11, unit11_props = find_peaks(-filt_g3, height=(0.2, 0.3), prominence=(0.25, 0.4),
                                  rel_height=0.3, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit11_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit11_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit11_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit11_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit11_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit11_props), x='widths', bins=50, ax=axs[1,2])

unit12, unit12_props = find_peaks(filt_g3, height=(0.05, 0.1), prominence=(0.14, 0.2),
                                  rel_height=0.22, width=(2, 38), distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit12_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit12_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit12_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit12_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit12_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit12_props), x='widths', bins=50, ax=axs[1,2])

unit14, unit14_props = find_peaks(filt_g3, height=(0.025, 0.04), prominence=(0.06, 0.1),
                                  rel_height=0.18, width=2, distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit14_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit14_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit14_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit14_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit14_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit14_props), x='widths', bins=50, ax=axs[1,2])

unit15, unit15_props = find_peaks(filt_g3, height=(0.07, 0.11), prominence=(0.19, 0.27),
                                  rel_height=0.22, width=(38, 110), distance=25)

# fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col')
# sns.stripplot(data=pd.DataFrame(unit15_props), x='peak_heights', ax=axs[0,0])
# sns.histplot(data=pd.DataFrame(unit15_props), x='peak_heights', bins=50, ax=axs[1,0])
# sns.stripplot(data=pd.DataFrame(unit15_props), x='prominences', ax=axs[0,1])
# sns.histplot(data=pd.DataFrame(unit15_props), x='prominences', bins=50, ax=axs[1,1])
# sns.stripplot(data=pd.DataFrame(unit15_props), x='widths', ax=axs[0,2])
# sns.histplot(data=pd.DataFrame(unit15_props), x='widths', bins=50, ax=axs[1,2])


# fig, ax = plt.subplots()
# ax.plot(time_25120801, filt_g3, c='k', lw=0.6)
# ax.scatter(time_25120801[unit11], filt_g3[unit11], color=color_dict_25120801[11])
# ax.scatter(time_25120801[unit12], filt_g3[unit12], color=color_dict_25120801[12])
# ax.scatter(time_25120801[unit14], filt_g3[unit14], color=color_dict_25120801[14])
# ax.scatter(time_25120801[unit15], filt_g3[unit15], color=color_dict_25120801[15])
# ax.scatter(time_25120801[unit15], unit15_props['width_heights'], marker='_', color=color_dict_25120801[15])

spike_idx_dict_25120801 = {
    1 : unit1,
    2 : unit2,
    3 : unit3,
    4 : unit4,
    5 : unit5,
    6 : unit6,
    7 : unit7,
    8 : unit8,
    11 : unit11,
    12 : unit12,
    14 : unit14,
    15 : unit15
    }

channels_25120801 = {1:0, 2:0, 3:0, 4:0, 5:0, 6:1, 7:1, 8:1, 11:2, 12:2, 14:2, 15:2}

unit_dict_25120801 = units_in_dp(channels_25120801) 

plot_spikes(arr_dict_25120801, time_25120801, spike_idx_dict_25120801, color_dict_25120801,
            unit_dict_25120801)

spike_freq_25120801 = spike_frequency(spike_idx_dict_25120801, time_25120801, 0.25)

filtered_freq_25120801 = filter_freq(spike_freq_25120801, 3)

plot_spike_freq(filtered_freq_25120801, color_dict_25120801)

crawling_start_25120801 = 140.3

#detección de la máxima frecuencia en el intervalo a ser estudiado para normalizar la frecuencia y visualizar en el mismo gráfico
selected_freq_25120801 = slice_frequency(filtered_freq_25120801, [[130, 790]])[0]

unit6_maxs_25120801, _ = find_peaks(selected_freq_25120801[6][1], height=10, prominence=10)
unit6_mins_25120801, _ = find_peaks(-selected_freq_25120801[6][1], prominence=10)
unit6_first_min = np.argmin(selected_freq_25120801[6][1][:unit6_maxs_25120801[0]])
unit6_mins_25120801 = np.insert(unit6_mins_25120801, 0, unit6_first_min)
del unit6_first_min


unit6_99_percent_point_25120801 = calculate_99_percent_points(selected_freq_25120801[6][1],
                                                              unit6_mins_25120801,
                                                              unit6_maxs_25120801)

ref_unit = 1 #unidad de referencia
interval_length = 3
correlation_pairs_25120801 = [(1,6), (6,11), (1,11)]


#normalizo cada ciclo entre 0 y 1, y los llevo al tamaño en puntos del ciclo más largo
norm_cycles_25120801 = normalize_cycles(selected_freq_25120801, unit6_99_percent_point_25120801, ref_unit)
          
three_cycle_intervals_25120801 = paste_cycles(norm_cycles_25120801, 3) #normalized_cycles, interval_length    

zscore_list_25120801 = []
for freq_dict in three_cycle_intervals_25120801:
    zscore_dict = {}
    for unit, norm_freq in freq_dict.items():   
        zscore_dict[unit] = zscore(norm_freq)
    
    zscore_list_25120801.append(zscore_dict)
    
xcorr_dict_25120801 = {}
for pair in correlation_pairs_25120801:
    xcorr_dict_25120801[pair] = []
    for zscore_dict in zscore_list_25120801:
        xcorr = (correlate(zscore_dict[pair[0]], zscore_dict[pair[1]]) 
                 / len(zscore_dict[pair[0]]))
        xcorr_dict_25120801[pair].append(xcorr)

x_axis_25120801 = np.linspace(0, 3, len(three_cycle_intervals_25120801[0][6]))
delta_x = x_axis_25120801[1]
norm_bin_fs = 1 / delta_x

#es el mismo arr_lag para todos porque el tiempo quedó normalizado
lag_arr_25120801 = correlation_lags(len(zscore_list_25120801[0][pair[0]]),
                                    len(zscore_list_25120801[0][pair[1]])) / norm_bin_fs

# #figura ventanas normalizadas
fig, axs = plt.subplots(nrows=6, ncols=len(three_cycle_intervals_25120801), sharey='row')
for i, unit in enumerate([1,6,11]):    
    for j, interval in enumerate(three_cycle_intervals_25120801):  
        axs[i, j].plot(x_axis_25120801,
                        interval[unit],
                        lw=1, c=color_dict_25120801[unit], label=unit_names[unit])
        
        axs[i, j].set_xticks([0,1,2,3])
        if i < 2:
            axs[i, j].set_xticklabels([])
        
    axs[i, len(three_cycle_intervals_25120801)-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

for i, pair in enumerate([(1,6), (6,11), (1,11)]):
    for j in range(len(three_cycle_intervals_25120801)):
        axs[i+3, j].plot(lag_arr_25120801, xcorr_dict_25120801[pair][j], lw=1, c=xcorr_colors[pair],
                          label=f'xcorr {xcorr_pair_names[pair]}')        
        axs[i+3, j].set_ylim(-1,1)
        axs[3, j].set_xticklabels([])
        axs[4, j].set_xticklabels([])
        axs[5, j].set_xticks([-3, 0, 3])
    axs[i+3, len(three_cycle_intervals_25120801)-1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")  


xcorr_6_11_25120801 = pd.DataFrame(xcorr_dict_25120801[(6, 11)], columns=np.round(lag_arr_25120801, 3))
xcorr_1_6_25120801 = pd.DataFrame(xcorr_dict_25120801[(1, 6)], columns=np.round(lag_arr_25120801, 3))
xcorr_1_11_25120801 = pd.DataFrame(xcorr_dict_25120801[(1, 11)], columns=np.round(lag_arr_25120801, 3))

new_x_ticks = np.arange(-3, 3.5, 0.5)

column_labels = xcorr_6_11_25120801.columns.astype(float).values

# Encontrar los índices de los valores más cercanos
closest_idxs = [np.abs(column_labels - value).argmin() for value in new_x_ticks]

plt.figure(figsize=(12,6))
sns.heatmap(xcorr_1_6_25120801.loc[:17, :], vmin=-1, vmax=1, cmap=custom_cmap)  
plt.xticks(closest_idxs, new_x_ticks, rotation=0)
plt.title('G1-G2 DE3 cross-correlations')
plt.tight_layout()

plt.figure(figsize=(12,6))
sns.heatmap(xcorr_6_11_25120801.loc[:17, :], vmin=-1, vmax=1, cmap=custom_cmap)  
plt.xticks(closest_idxs, new_x_ticks, rotation=0)
plt.title('G2-G3 DE3 cross-correlations')
plt.tight_layout()

plt.figure(figsize=(12,6))
sns.heatmap(xcorr_1_11_25120801.loc[:17, :], vmin=-1, vmax=1, cmap=custom_cmap)  
plt.xticks(closest_idxs, new_x_ticks, rotation=0)
plt.title('G1-G3 DE3 cross-correlations')
plt.tight_layout()

max_xcorr_peaks_1_6_25120801 = max_peak_in_section(lag_arr = lag_arr_25120801,
                                                   xcorr_list = xcorr_dict_25120801[(1, 6)][:18],
                                                   max_lag = 0.7,
                                                   min_xcorr_value = 0.25)

max_xcorr_peaks_1_6_25120801['rad_lag'] = max_xcorr_peaks_1_6_25120801['lag'] * 2 * np.pi

max_xcorr_peaks_6_11_25120801 = max_peak_in_section(lag_arr = lag_arr_25120801,
                                                    xcorr_list = xcorr_dict_25120801[(6, 11)][:18],
                                                    max_lag = 0.7,
                                                    min_xcorr_value = 0.25)

max_xcorr_peaks_6_11_25120801['rad_lag'] = max_xcorr_peaks_6_11_25120801['lag'] * 2 * np.pi

mean_res_len_6_11_25120801 = (
    max_xcorr_peaks_6_11_25120801["rad_lag"]
    .rolling(window=8)
    .apply(mean_resultant_length, raw=True)
)

window_end_idx_25120801 = mean_res_len_6_11_25120801.loc[7:].idxmax()

window_start_idx_25120801 = window_end_idx_25120801 - 7

max_xcorr_peaks_1_11_25120801 = max_peak_in_section(lag_arr = lag_arr_25120801,
                                                   xcorr_list = xcorr_dict_25120801[(1, 11)][:18],
                                                   max_lag = 2 * 0.7,
                                                   min_xcorr_value = 0.25)

max_xcorr_peaks_1_11_25120801['rad_lag'] = max_xcorr_peaks_1_11_25120801['lag'] * 2 * np.pi

xcorr_peaks_1_6_25120801 = max_xcorr_peaks_1_6_25120801.loc[window_start_idx_25120801:window_end_idx_25120801, :]
xcorr_peaks_1_6_25120801.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_1_6_25120801.insert(1, 'interval', xcorr_peaks_1_6_25120801.index.values + 1)
xcorr_peaks_1_6_25120801.insert(2, 'ganglion_pair', 'G1-G2')

xcorr_peaks_6_11_25120801 = max_xcorr_peaks_6_11_25120801.loc[window_start_idx_25120801:window_end_idx_25120801, :]
xcorr_peaks_6_11_25120801.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_6_11_25120801.insert(1, 'interval', xcorr_peaks_6_11_25120801.index.values + 1)
xcorr_peaks_6_11_25120801.insert(2, 'ganglion_pair', 'G2-G3')

xcorr_peaks_1_11_25120801 = max_xcorr_peaks_1_11_25120801.loc[window_start_idx_25120801:window_end_idx_25120801, :]
xcorr_peaks_1_11_25120801.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_1_11_25120801.insert(1, 'interval', xcorr_peaks_1_11_25120801.index.values + 1)
xcorr_peaks_1_11_25120801.insert(2, 'ganglion_pair', 'G1-G3')

xcorr_peaks_25120801 = pd.concat([xcorr_peaks_1_6_25120801, xcorr_peaks_6_11_25120801, xcorr_peaks_1_11_25120801], ignore_index=True)


start = selected_freq_25120801[6][0][unit6_99_percent_point_25120801[window_start_idx_25120801]]
end = selected_freq_25120801[6][0][unit6_99_percent_point_25120801[window_end_idx_25120801 + 4]]

stable_region_25120801 = slice_frequency(selected_freq_25120801, [[start, end]])[0]

cycle_limits_25120801 = (unit6_99_percent_point_25120801[window_start_idx_25120801:window_end_idx_25120801 + 3 + 1]
                         - unit6_99_percent_point_25120801[window_start_idx_25120801]) #uno más porque excluye al último

stable_unit6_maxs_25120801, _ = find_peaks(stable_region_25120801[6][1], height=10, prominence=10)
stable_unit6_mins_25120801, _ = find_peaks(-stable_region_25120801[6][1], prominence=10)
unit6_first_min = np.argmin(stable_region_25120801[6][1][:stable_unit6_maxs_25120801[0]])
stable_unit6_mins_25120801 = np.insert(stable_unit6_mins_25120801, 0, unit6_first_min)
del unit6_first_min

stable_unit11_maxs_25120801, _ = find_peaks(stable_region_25120801[11][1], height=10, prominence=10)
stable_unit11_mins_25120801, _ = find_peaks(-stable_region_25120801[11][1], prominence=10)
unit11_first_min = np.argmin(stable_region_25120801[11][1][:stable_unit11_maxs_25120801[0]])
stable_unit11_mins_25120801 = np.insert(stable_unit11_mins_25120801, 0, unit11_first_min)
del unit11_first_min

stable_unit1_maxs_25120801, _ = find_peaks(stable_region_25120801[1][1], height=8, prominence=8)
stable_unit1_mins_25120801, _ = find_peaks(-stable_region_25120801[1][1], prominence=8)


stable_max_dict_25120801 = {
    1 : stable_unit1_maxs_25120801,
    6 : stable_unit6_maxs_25120801,
    11 : stable_unit11_maxs_25120801
    }


stable_min_dict_25120801 = {
    1 : stable_unit1_mins_25120801,
    6 : stable_unit6_mins_25120801,
    11 : stable_unit11_mins_25120801
    }

ax_unit_dict_25120801 = {0:1, 1:6, 2:11} 

fig, axes = plt.subplots(nrows=3, sharex=True)
for idx, unit in ax_unit_dict_25120801.items():
    axes[idx].plot(stable_region_25120801[unit][0], stable_region_25120801[unit][1], c=color_dict_25120801[unit])
    
    axes[idx].scatter(stable_region_25120801[unit][0][stable_max_dict_25120801[unit]],
                      stable_region_25120801[unit][1][stable_max_dict_25120801[unit]],
                      c='red')
    axes[idx].scatter(stable_region_25120801[unit][0][stable_min_dict_25120801[unit]],
                      stable_region_25120801[unit][1][stable_min_dict_25120801[unit]],
                      c='dodgerblue')
for i, ax in enumerate(axes):
    for limit in cycle_limits_25120801:
        ax.axvline(x=stable_region_25120801[11][0][limit],
                    lw=1, c='grey', ls='-')


local_unit_g2 = 6
local_de3_props_25120801 = local_de3_parameters(stable_region_25120801,
                                                cycle_limits_25120801, 
                                                stable_min_dict_25120801,
                                                stable_max_dict_25120801,
                                                local_unit_g2)

de3_units_25120801 = [1,6,11]

spike_freq_amp_25120801 = spike_freq_amp_in_chain(stable_region_25120801,
                                                  cycle_limits_25120801, 
                                                  stable_min_dict_25120801,
                                                  stable_max_dict_25120801,
                                                  de3_units_25120801)

#zona de anticorrelación más estable

min_xcorr_peaks_1_6_25120801 = min_peak_in_section(lag_arr = lag_arr_25120801,
                                                   xcorr_list = xcorr_dict_25120801[(1, 6)][:18],
                                                   max_lag = 0.7,
                                                   min_xcorr_value = 0.25)

min_xcorr_peaks_1_6_25120801['rad_lag'] = min_xcorr_peaks_1_6_25120801['lag'] * 2 * np.pi

min_xcorr_peaks_1_6_25120801['xcorr'] = min_xcorr_peaks_1_6_25120801['xcorr'] * -1

min_xcorr_peaks_6_11_25120801 = min_peak_in_section(lag_arr = lag_arr_25120801,
                                                    xcorr_list = xcorr_dict_25120801[(6, 11)][:18],
                                                    max_lag = 0.7,
                                                    min_xcorr_value = 0.25)

min_xcorr_peaks_6_11_25120801['rad_lag'] = min_xcorr_peaks_6_11_25120801['lag'] * 2 * np.pi

min_xcorr_peaks_6_11_25120801['xcorr'] = min_xcorr_peaks_6_11_25120801['xcorr'] * -1

mean_res_len_6_11_25120801_min = (
    min_xcorr_peaks_6_11_25120801["rad_lag"]
    .rolling(window=8)
    .apply(mean_resultant_length, raw=True)
)

window_end_idx_25120801_min = mean_res_len_6_11_25120801_min.loc[7:].idxmax()

window_start_idx_25120801_min = window_end_idx_25120801_min - 7

min_xcorr_peaks_1_11_25120801 = min_peak_in_section(lag_arr = lag_arr_25120801,
                                                   xcorr_list = xcorr_dict_25120801[(1, 11)][:18],
                                                   max_lag = 2 * 0.7,
                                                   min_xcorr_value = 0.25)

min_xcorr_peaks_1_11_25120801['rad_lag'] = min_xcorr_peaks_1_11_25120801['lag'] * 2 * np.pi

min_xcorr_peaks_1_11_25120801['xcorr'] = min_xcorr_peaks_1_11_25120801['xcorr'] * -1

xcorr_peaks_1_6_25120801_min = min_xcorr_peaks_1_6_25120801.loc[window_start_idx_25120801_min:window_end_idx_25120801_min, :]
xcorr_peaks_1_6_25120801_min.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_1_6_25120801_min.insert(1, 'interval', xcorr_peaks_1_6_25120801_min.index.values + 1)
xcorr_peaks_1_6_25120801_min.insert(2, 'ganglion_pair', 'G1-G2')

xcorr_peaks_6_11_25120801_min = min_xcorr_peaks_6_11_25120801.loc[window_start_idx_25120801_min:window_end_idx_25120801_min, :]
xcorr_peaks_6_11_25120801_min.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_6_11_25120801_min.insert(1, 'interval', xcorr_peaks_6_11_25120801_min.index.values + 1)
xcorr_peaks_6_11_25120801_min.insert(2, 'ganglion_pair', 'G2-G3')

xcorr_peaks_1_11_25120801_min = min_xcorr_peaks_1_11_25120801.loc[window_start_idx_25120801_min:window_end_idx_25120801_min, :]
xcorr_peaks_1_11_25120801_min.insert(0, 'Exp_ID', '25120801')
xcorr_peaks_1_11_25120801_min.insert(1, 'interval', xcorr_peaks_1_11_25120801_min.index.values + 1)
xcorr_peaks_1_11_25120801_min.insert(2, 'ganglion_pair', 'G1-G3')

xcorr_peaks_25120801_min = pd.concat([xcorr_peaks_1_6_25120801_min, xcorr_peaks_6_11_25120801_min, xcorr_peaks_1_11_25120801_min], ignore_index=True)

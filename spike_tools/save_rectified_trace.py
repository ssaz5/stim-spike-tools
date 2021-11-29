# sys.path.insert(0, '../load_intan_rhd_format/load_intan_rhd_format/')

import numpy as np
import joblib
import scipy.io as sio
import pandas as pd
import os
from pathlib import Path

from utils.intanutils import read_amplifier
from utils.filter import bandpass_filter, notch_filter, lowpass_filter
from utils import find_nearest

from utils.spikeutils import remove_artefacts_mod

import matplotlib.pyplot as plt
import seaborn as sns


import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')


def moving_average(x, w):
    w = int(w)
    return np.convolve(x, np.ones(w)/w, 'same')

import argparse
parser = argparse.ArgumentParser(description='Save Rectified Trace.')

parser.add_argument('-ch',  '--channel_id', type=int, default=0)
parser.add_argument('-fn', '--file_name', type=str, default='oleo_stimulation_210813_134444/')
parser.add_argument('-d', '--date', type=str, default='')
parser.add_argument('-pp',  '--pulse_period', type=int, default=4000)
parser.add_argument('-np',  '--num_pulses', type=int, default=10)
parser.add_argument('-au', '--art_time_usec', type=int, default = 1200)

args = parser.parse_args()


date = args.date
if not date:
    date = config['Experiment Information']['date'] 


### Create Channel List ###
channel_letters = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
channel_numbers = np.arange(32)


all_channels = np.array(np.meshgrid( channel_numbers, channel_letters))
all_channels = np.core.defchararray.add(np.core.defchararray.add(all_channels[1],'-'),np.char.zfill(all_channels[0], 3))
all_channels = all_channels.flatten()

n_channels = len(all_channels)

    

### Set variables ###

pulse_width_usec=args.pulse_period
num_pulses=args.num_pulses




channel_num = args.channel_id
recording_channel = all_channels[channel_num].capitalize()

art_time_usec = args.art_time_usec

f_low = int(config['Filtering']['fLow'])
f_high = int(config['Filtering']['fHigh'])
samplingFrequency = int(config['Filtering']['fSampling'])

    

## Define Paths ##
dataPath = config['Paths']['home_dir']+'projects/'+config['Experiment Information']['name']+"/monkeys/oleo/"
procDataPath = dataPath+'intanproc'
rawDataPath =  dataPath+'intanraw'
rectDataPath = dataPath+'recttrace'




artefact_times_path = os.path.join(rawDataPath, date, 'artefact_time_'+date+'.pkl')
samp_filename = os.path.join(rawDataPath,date,'samp_time_'+date+'.pkl')



#### Get names of all directories with the specified 'date'.

with os.scandir(rawDataPath) as it:
    dirs = [entry.name for entry in it if (entry.is_dir() and entry.name.find(date) != -1 and entry.name != date)]
dirs.sort()

d = dirs[0]
directory_path = os.path.join(rawDataPath, d)
# Get all raw neural data files
with os.scandir(os.path.join(rawDataPath, d)) as it:
    files = [entry.name for entry in it if (entry.is_file() and entry.name.find('amp') != -1)]
files.sort()  # The files are randomly loaded, so sort them
print(n_channels, len(files))
assert len(files) == n_channels  # Check if number of files matches number of channels



rectFilename = 'final_trace_channel_'+all_channels[channel_num]+'.pkl'
rectFilePath = os.path.join(rectDataPath, d)

### Load MWORKS DATA ###

mworksDirproc = dataPath+'mworksproc/'
mworksFilename = [i for i in os.listdir(mworksDirproc) if date in i][0]
data_info = pd.read_csv(os.path.join(mworksDirproc, mworksFilename))
samp_on_id = data_info.stim_id.values
samp_on_current = data_info.stim_current.values


### Load Artefact Data ###
artefact_times = joblib.load(artefact_times_path)
nan_sums = np.sum(np.isnan(artefact_times), axis=0)
good_ones = np.where(nan_sums < 96)[0]

artefact_times = np.nanmedian(artefact_times, axis=0).astype(int)
artefact_times = artefact_times[good_ones]


if 1: #not os.path.exists(os.path.join(rectFilePath, rectFilename)):
    print('Calculating v1 and v2')

    v = read_amplifier(os.path.join(rawDataPath, d, files[channel_num]))  # In microvolts



    # ### Load Samp On File ####

    # if os.path.exists(samp_filename):
    #     samp_on = joblib.load(samp_filename)
    # else:
    #     filename = Path(os.path.join(directory_path, 'board-DIGITAL-IN-02.dat'))
    #     fid = open(filename, 'r')
    #     filesize = os.path.getsize(filename) # in bytes
    #     num_samples = filesize // 2 # uint16 = 2 bytes
    #     din02 = np.fromfile(fid, 'uint16', num_samples)
    #     fid.close()

    #     samp_on, = np.nonzero(din02[:-1] < din02[1:]) # Look for 0->1 transitions
    #     samp_on = samp_on + 1 # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    #     joblib.dump(samp_on, samp_filename)




    ### Remove Artefacts ###
    v1 = remove_artefacts_mod(v, artefact_times, samplingFrequency, f_low, f_high, art_time_usec=art_time_usec, 
                         v_thres=400, num_pulses=num_pulses, pulse_width_usec=pulse_width_usec, apply_salpa=False, samp_current=samp_on_current).astype(np.float16);

    # print(type(v1))
    # print(v1.dtype)
    # exit()

    v2 = remove_artefacts_mod(v, artefact_times, samplingFrequency, f_low, f_high, art_time_usec=art_time_usec, 
                         v_thres=2000, num_pulses=num_pulses, pulse_width_usec=pulse_width_usec, apply_salpa=True, samp_current=samp_on_current).astype(np.float16);


    ### Save Data ###


    data_dict = {'v1': v1, 'v2': v2}


    print(rectFilePath)

    os.makedirs(rectFilePath, exist_ok = True)

    joblib.dump(data_dict, os.path.join(rectFilePath, rectFilename), compress=3)
else:
    print('Skipping v1 v2 calculation')
    v = read_amplifier(os.path.join(rawDataPath, d, files[channel_num]))  # In microvolts
    data_dict = joblib.load(os.path.join(rectFilePath, rectFilename))
    v1 = data_dict['v1']
    v2 = data_dict['v2']


### Save Plots ###

unique_currents = np.unique(samp_on_current)
unique_ids = np.unique(samp_on_id)

rectFilePath

for stim_id in unique_ids:
    fig, axes = plt.subplots(1, len(unique_currents), sharex=True, sharey=True, figsize=[25,5])
    print('Processing Stimulation Channel: ', stim_id)
    average_signals = {}
    for i, stim_current in enumerate(unique_currents):


        pre_ = 1000
        post_ = 2000
        
        average_signal = np.zeros([1,pre_+post_])
        count = 0
        color_ = 'm'

        for rep_num, art_num in enumerate(np.where(np.logical_and(data_info['stim_id'] == stim_id , data_info['stim_current'] == stim_current))[0]):
            f_low = 300
            f_high = 5000
#             broadband_signal_bandpass_rectification = np.maximum(0, bandpass_filter(v2[artefact_times[art_num]-pre_:artefact_times[art_num]+post_], samplingFrequency, f_low, f_high))
#             broadband_signal_bandpass_rectification = np.abs(bandpass_filter(v2[artefact_times[art_num]-pre_:artefact_times[art_num]+post_], samplingFrequency, f_low, f_high))
            broadband_signal_bandpass_rectification = bandpass_filter(v2[artefact_times[art_num]-pre_:artefact_times[art_num]+post_], samplingFrequency, f_low, f_high)
#             f_low = 0.1
            f_low = 200
            broadband_signal_low_pass = lowpass_filter(broadband_signal_bandpass_rectification, samplingFrequency, f_low)
#             broadband_signal_low_pass = moving_average(broadband_signal_bandpass_rectification, 100)
            average_signal += broadband_signal_low_pass
            count +=1

        average_signal = average_signal/count
        average_signals[stim_current] = average_signal
        
        t= 1000* np.arange(-pre_, post_) /samplingFrequency
        plt.subplot(1,len(unique_currents), i+1)
    #     ax = 
        plt.plot(t, average_signal.flatten(), color_)
        plt.title('Current = '+str(stim_current))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (uV)')
        plt.ylim([-5,55])


    plt.suptitle('Stim_channel: '+stim_id.capitalize()+ '; Recording Channel: '+all_channels[channel_num].capitalize())
    plot_dir = os.path.join(rectFilePath , 'Figures', stim_id)
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = 'trace_recording_site_'+all_channels[channel_num]+'.png'
    print(plot_filename)
    plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches='tight')
    plot_data_dir = os.path.join(rectFilePath , 'data', stim_id)
    os.makedirs(plot_data_dir, exist_ok=True)
    plot_data_filename = 'trace_recording_site_'+all_channels[channel_num]+'.pkl'
    print(plot_data_filename)
    joblib.dump(average_signals, os.path.join(plot_data_dir, plot_data_filename))
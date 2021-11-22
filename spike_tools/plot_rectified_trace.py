# sys.path.insert(0, '../load_intan_rhd_format/load_intan_rhd_format/')

import numpy as np
import joblib
import scipy.io as sio
import pandas as pd
import os
from pathlib import Path

from utils.intanutils import read_amplifier
from utils.filter import bandpass_filter, notch_filter
from utils import find_nearest

from utils.spikeutils import remove_artefacts_mod

import matplotlib.pyplot as plt
import seaborn as sns

import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')


import argparse
parser = argparse.ArgumentParser(description='Run spike detection.')

parser.add_argument('-ch',  '--channel_id', type=int, default=0)
parser.add_argument('-fn', '--file_name', type=str, default='oleo_stimulation_210813_134444/')
parser.add_argument('-d', '--date', type=str, default='')
parser.add_argument('-pp',  '--pulse_period', type=int, default=4000)
parser.add_argument('-np',  '--num_pulses', type=int, default=10)


args = parser.parse_args()


date = args.date
if not date:
    date = config['Experiment Information']['date'] 

## Define Paths ##
dataPath = config['Paths']['home_dir']+'projects/'+config['Experiment Information']['name']+"/monkeys/oleo/"
procDataPath = dataPath+'intanproc'
rawDataPath =  dataPath+'intanraw'
rectDataPath = dataPath+'recttrace'


artefact_times_path = os.path.join(rawDataPath, date, 'artefact_time_'+date+'.pkl')
samp_filename = os.path.join(rawDataPath,date,'samp_time_'+date+'.pkl')



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




# art_time_usec = args.art_time_usec

f_low = int(config['Filtering']['fLow'])
f_high = int(config['Filtering']['fHigh'])
samplingFrequency = int(config['Filtering']['fSampling'])

### Load Samp On File ####

if os.path.exists(samp_filename):
    samp_on = joblib.load(samp_filename)
else:
    filename = Path(os.path.join(directory_path, 'board-DIGITAL-IN-02.dat'))
    fid = open(filename, 'r')
    filesize = os.path.getsize(filename) # in bytes
    num_samples = filesize // 2 # uint16 = 2 bytes
    din02 = np.fromfile(fid, 'uint16', num_samples)
    fid.close()

    samp_on, = np.nonzero(din02[:-1] < din02[1:]) # Look for 0->1 transitions
    samp_on = samp_on + 1 # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    joblib.dump(samp_on, samp_filename)

    
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

#### PLOTTING VARIABLES ######

num_rows = 16


### Make Plots Data ###

with os.scandir(rawDataPath) as it:
    dirs = [entry.name for entry in it if (entry.is_dir() and entry.name.find(date) != -1 and entry.name != date)]
dirs.sort()

d = dirs[0]
directory_path = os.path.join(rawDataPath, d)


rectFilePath = os.path.join(rectDataPath, d)
plot_dir = os.path.join(rectFilePath , 'Figures')
os.makedirs(plot_dir, exist_ok=True)

print(rectFilePath)


unique_currents = np.unique(samp_on_current)
unique_ids = np.unique(samp_on_id)
stim_id = unique_ids[args.channel_id]

print('Stimulation site: ',stim_id)



for channel_num, recording_channel in enumerate(all_channels):

    rectFilename = 'final_trace_channel_'+recording_channel+'.pkl'

    data_dict = joblib.load(os.path.join(rectFilePath, rectFilename))
    v1 = data_dict['v1']
    v2 = data_dict['v2']
    
    if not (channel_num % num_rows):
        fig, axes = plt.subplots(num_rows, len(unique_currents), sharex=True, sharey=True, figsize=[25,int(num_rows*5)])
        print(channel_num)
    # fig = plt.figure(num = fig, figsize=[25, 5])
    for i, stim_current in enumerate(unique_currents):


        pre_ = 1000
        post_ = 1500

        average_signal = np.zeros([1,pre_+post_])
        count = 0
        color_ = 'm'

        for rep_num, art_num in enumerate(np.where(np.logical_and(data_info['stim_id'] == stim_id , data_info['stim_current'] == stim_current))[0]):
            f_low = 1000
            f_high = 5000
            broadband_signal_bandpass_rectification = np.maximum(0, bandpass_filter(v2[artefact_times[art_num]-pre_:artefact_times[art_num]+post_], samplingFrequency, f_low, f_high))
            f_low = 0.1
            f_high = 200
            broadband_signal_low_pass = bandpass_filter(broadband_signal_bandpass_rectification, samplingFrequency, f_low, f_high)
            average_signal += broadband_signal_low_pass
            count +=1
        
        average_signal = average_signal/count

        t= 1000* np.arange(-pre_, post_) /samplingFrequency
        plt.subplot(channel_num % num_rows + 1,len(unique_currents), i+1)
        
        plt.plot(t, broadband_signal_low_pass, color_)
        plt.title('Current = '+str(stim_current))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (uV)')
        
    if not (channel_num+1) % num_rows:
        plt.suptitle('Stim_channel: '+stim_id.capitalize()+ '; Recording Channels: '+all_channels[channel_num-num_rows].capitalize() + ' to '+ all_channels[channel_num].capitalize())
        plot_dir = os.path.join(rectFilePath , 'Figures/Grouped/', stim_id)
        os.makedirs(plot_dir, exist_ok=True)
        plot_filename = 'trace_recording_site_'+all_channels[channel_num-num_rows].capitalize() + '_to_'+ all_channels[channel_num].capitalize()+'.png'
        print(plot_filename)
        plt.savefig(os.path.join(plot_dir, plot_filename), bbox_inches='tight')
        
    

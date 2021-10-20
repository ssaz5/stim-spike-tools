from __future__ import division, print_function, unicode_literals


import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from pathlib import Path

import joblib


import sqlite3
import zlib

import msgpack

from scipy.signal import find_peaks

from utils.mworksutils import get_trial_indices

try:
    buffer
except NameError:
    # Python 3
    buffer = bytes

def moving_average(x, w):
    w = int(w)
    return np.convolve(x, np.ones(w), 'same')

class MWK2Reader(object):

    _compressed_text_type_code = 1
    _compressed_msgpack_stream_type_code = 2

    def __init__(self, filename):
        self._conn = sqlite3.connect(filename)
        self._unpacker = msgpack.Unpacker(raw=False, strict_map_key=False)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def _decompress(data):
        return zlib.decompress(data, -15)

    def __iter__(self):
        for code, time, data in self._conn.execute('SELECT * FROM events'):
            if not isinstance(data, buffer):
                yield (code, time, data)
            else:
                try:
                    obj = msgpack.unpackb(data, raw=False)
                except msgpack.ExtraData:
                    # Multiple values, so not valid compressed data
                    pass
                else:
                    if isinstance(obj, msgpack.ExtType):
                        if obj.code == self._compressed_text_type_code:
                            yield (code,
                                   time,
                                   self._decompress(obj.data).decode('utf-8'))
                            continue
                        elif (obj.code ==
                              self._compressed_msgpack_stream_type_code):
                            data = self._decompress(obj.data)
                self._unpacker.feed(data)
                try:
                    while True:
                        yield (code, time, self._unpacker.unpack())
                except msgpack.OutOfData:
                    pass

                
def apply_bandpass(data, fs, flow, fhigh):
    wl = flow / (fs / 2.)
    wh = fhigh / (fs / 2.)
    wn = [wl, wh]

    # Designs a 2nd-order Elliptic band-pass filter which passes
    # frequencies between 0.03 and 0.6, an with 0.1 dB of ripple
    # in the passband, and 40 dB of attenuation in the stopband.
    # The question is, do we really want to use IIR filter design?
    # Isn't it the case that IIR filters introduce refractory period
    # artifacts, and thus FIRs are preferred in practice?
    b, a = signal.ellip(2, 0.1, 40, wn, 'bandpass', analog=False)
    # To match Matlab output, we change default padlen from
    # 3*(max(len(a), len(b))) to 3*(max(len(a), len(b)) - 1)
    return signal.filtfilt(b, a, data, padlen=3*(max(len(a),len(b))-1))


# sys.path.insert(0, '/Users/etotheipiplusone/Dropbox (MIT)/load_intan_rhd_format/')
sys.path.insert(0, '../load_intan_rhd_format/load_intan_rhd_format/')

from load_intan_rhd_format import read_data

import argparse
import configparser


config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')


parser = argparse.ArgumentParser(description='Run spike detection.')


parser.add_argument('-ch',  '--channel_id', type=int, default=0)
parser.add_argument('-fn', '--file_name', type=str, default='oleo_stimulation_210813_134444/')
parser.add_argument('-m',  '--get_median', type=int, default=0)
parser.add_argument('-p', '--prefix', type=str, default='')
parser.add_argument('-pp',  '--pulse_period', type=int, default=4000)
parser.add_argument('-np',  '--num_pulses', type=int, default=10)
parser.add_argument('-d', '--date', type=str, default='')


args = parser.parse_args()

def main():
    rawDataPath = "/braintree/data2/active/users/ssazaidi/projects/"+config['Experiment Information']['name']+"/monkeys/oleo/"
    
    if args.date == '':
        file_name = args.file_name
        date = file_name.split('_')[-2]
    else:
        date = args.date
        print(os.path.join(rawDataPath,'intanraw'))
        file_name = [i for i in os.listdir(os.path.join(rawDataPath,'intanraw')) if date in i and i != date][0]
        
    
    directory_path = os.path.join(rawDataPath, "intanraw/", file_name)

    
    prefix = args.prefix
    os.makedirs(rawDataPath+"intanraw/"+prefix+date, exist_ok = True)


    samplingFrequency = 20000
    pulse_period_ms = int(args.pulse_period/1000) # milliseconds
    num_pulses = args.num_pulses
    sum_window = int((pulse_period_ms* num_pulses * samplingFrequency)/1000)
    
    
    pre_ = sum_window
    post_ = 2*sum_window
    length_ = pre_+post_
    half_ = length_/(2*samplingFrequency)
    
    

    art_width_usec = 340
    art_width = np.ceil(art_width_usec*samplingFrequency/1e6).astype(int)
    
    pulse_train = np.zeros([sum_window,])
    pulse_train[np.concatenate([i + np.arange(art_width) for i in np.arange(0,sum_window,int(sum_window/num_pulses))+1])] = 1
    pulse_train = pulse_train[::-1]
    

    ################ SAMP_ON DIGITAL LINE 2 #####################################
        
    samp_filename = rawDataPath+"intanraw/"+prefix+date+'/samp_time_'+date+'.pkl'

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

        
    ############### LOAD MWORKS DATA ################################    
    data_file_name = os.path.join(rawDataPath, 'intanraw',date, 'all_data.pkl')
    print(data_file_name)
    if os.path.exists(data_file_name):
        mworks_data = joblib.load(data_file_name)
    else:
        print('MWORKS RAW PKL DOESN\'T EXIST. Please run mwk_create.py')
        return
    # event_file.close()

    
    ################# Prepare current values to check for control ###########################
    
    current_events = mworks_data.loc[mworks_data['name'] == 'stim_current'].reset_index(drop=True)
    id_events = mworks_data.loc[mworks_data['name'] == 'stim_id'].reset_index(drop=True)
    current_times = np.array([row.time for i, row in current_events.iterrows()])
    id_times = np.array([row.time for i, row in id_events.iterrows()])

    current_trials = get_trial_indices(current_events, df=True, delay_sec=0.5)
    id_trials = get_trial_indices(id_events, df=True, delay_sec = 0.5)


    correct_id = []
    correct_current = []

    curr_idx = 0

    while np.all(np.array(current_events.iloc[current_trials[curr_idx]].data) == 1):
        curr_idx += 1

    for i in range(len(id_trials)):
        current_trial = current_trials[curr_idx]
        id_trial = id_trials[i]

        if '' in np.array(id_events.iloc[id_trial].data):
            continue

        if len(id_trial) > 8:
            id_trial = id_trial[-8:]


        for idx, j in enumerate(id_trial):
            try:
                correct_current.append(current_trial[idx])
                correct_id.append(j)
            except:
                pass

        if len(id_trial) > 1:
            curr_idx += 1

    samp_on_id = np.array(id_events.iloc[correct_id].data)[:len(samp_on)]

    samp_on_current = np.array(current_events.iloc[correct_current].data)[:len(samp_on)]
    
    ###################################################################
    
    channel_letters = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    channel_numbers = np.arange(32)


    all_channels = np.array(np.meshgrid( channel_numbers, channel_letters))
    all_channels = np.core.defchararray.add(np.core.defchararray.add(all_channels[1],'-'),np.char.zfill(all_channels[0], 3))
    all_channels = all_channels.flatten()

    missing_channels = []

    if args.get_median:
        artefact_files = [os.path.join('artefact_delays_'+date+'_'+i+'.pkl') for i in all_channels]
        artefact_times = np.zeros([len(artefact_files), len(samp_on)]) 


        for i, artefact_file in enumerate(artefact_files):
            try:
                artefact_info = joblib.load(rawDataPath+"intanraw/"+os.path.join(prefix+date, artefact_file))
                artefact_times[i,:] = artefact_info['artefact_delay']

            except:
                print('Data Missing for channel ', artefact_file, i)
                missing_channels.append(i)

        if missing_channels:
            print('Get minimum artefact delays for: ', missing_channels)

        else:
#             artefact_times =  np.nanmedian(artefact_times, axis=0) # + samp_on 
            joblib.dump(artefact_times, os.path.join(rawDataPath+"intanraw/"+prefix+date, 'artefact_time_'+date+'.pkl'))


    else:

        channel_name = all_channels[args.channel_id]
        print(channel_name )

        artefact_delay_filename = rawDataPath+"intanraw/"+prefix+date+'/artefact_delays_'+date+'_'+channel_name+'.pkl'
        
        if os.path.exists(artefact_delay_filename):
            return

        channel_name = channel_name.upper()

        filename = os.path.join(directory_path , 'amp-'+channel_name+'.dat') # amplifier channel data
        # filename = 'oleo_normalizers_210208_134001/amp-C-022.dat'
        fid = open(filename, 'r')
        filesize = os.path.getsize(filename) # in bytes
        num_samples = filesize // 2 # int16 = 2 bytes
        v = np.fromfile(fid, 'int16', num_samples)
        fid.close()


        v = v * 0.195 # convert to microvolts

        sub_samp_on = []
        artefact_on = []
        sub_samp_number = []


        failed_samp_on = []
        failed_stim_number = []


        v_thres = 400

        shift = 0

        sub_signals = np.zeros([len(samp_on), length_])

        for idx, s in enumerate(samp_on):
            i = idx
            i +=shift
            vthres= v_thres
            sub_signal = v[samp_on[i]-pre_: samp_on[i]+post_]
            band_pass_sub_signal = apply_bandpass(sub_signal, samplingFrequency, 300,6000)
            sub_signals[idx,:] = sub_signal
            peaks= np.array([])
            while peaks.size < 10 :
                peaks = find_peaks(np.abs(band_pass_sub_signal), vthres)[0]
                vthres = vthres*0.9
            print(i, peaks, type(peaks), vthres)
            temp = np.zeros_like(sub_signal)


            try:
                if samp_on_current[i] > 0:
        #             artefact_time = np.min(peaks)
                    ########### SULE EDITS -- GET BEGINNING OF PULSE TRAIN #########################
                    if vthres < 100:
                        print('This did not work!')
                        raise Exception("Let's not!")

                    temp[peaks] = 1

                    sig_av = np.convolve(np.square (band_pass_sub_signal), pulse_train, 'valid')
                    art_start = np.argmax(sig_av) - art_width

                     ########### SULE EDITS -- GET BEGINNING OF PULSE TRAIN #########################
                else:
                    art_start = 0
                        
                artefact_time =  art_start
                artefact_on.append(artefact_time)
                sub_samp_on.append(s)
                sub_samp_number.append(idx)
            except:
        #         print('Missing stim on: ', s)
                failed_samp_on.append(s)
                failed_stim_number.append(idx)
                pass



        samp_on_ = np.array(samp_on) * 1000 / samplingFrequency


        sub_samp_on_ = np.array(sub_samp_on) * 1000 / samplingFrequency


        artefact_on_ = 1000*(np.array(artefact_on)/samplingFrequency - half_)



        artefact_delay = np.zeros_like(samp_on, dtype=float)

        artefact_delay[sub_samp_number] = np.array(artefact_on) - pre_ + np.array(sub_samp_on)

        artefact_delay[failed_stim_number] = np.nan

        artefact_info= {'artefact_delay':artefact_delay}


        joblib.dump(artefact_info, artefact_delay_filename)
        
        return

if __name__ == "__main__":
    main()
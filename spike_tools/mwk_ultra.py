import os
import sys
sys.path.insert(0, '/Library/Application Support/MWorks/Scripting/Python')
# sys.path.insert(0, 'Scripting/Python')

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import scipy.io as sio

from pathlib import Path

import joblib

# from mworks.data import MWKFile

from utils.mworksutils import MWK2Reader
from utils.mworksutils import get_trial_indices

import argparse
import configparser
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')



parser = argparse.ArgumentParser(description='Run Mworks Analysis.')
parser.add_argument('mworks_file', type=str,
    help='Mworks file Name: To be changed soon', nargs='?',default='')
parser.add_argument('photodiode_file', type=str,
    help='Photodiode file Name: To be changed soon', nargs='?',default='-')
parser.add_argument('samp_on_file', type=str,
    help='sample on file Name: To be changed soon', nargs='?',default='')
parser.add_argument('--remove_artefacts', type=int, nargs='?',default=0)
parser.add_argument('--session_num', type=int,default=0)
parser.add_argument('--date', type=str)
args = parser.parse_args()
date = args.date
if not date:
    date = config['Experiment Information']['date']


SAMPLING_FREQUENCY_HZ = 20000  # Intan recording controller sampling frequency (to convert time units to ms)
THRESHOLD = 0.9  # Threshold for detecting the first rising edge in the oscillating photodiode signal



def equal_for_all_trials(events):
    return all(e.data == events[0].data for e in events)


def listify_events(events):
    return list(e.data for e in events)

def get_events(filepath, names, time_range=[0, np.inf]):
    data_dict = {
        'code': [],
        'name': [],
        'time': [],
        'data': [],
    }
    
    with MWK2Reader(filepath) as event_file:
        code_to_name, name_to_code = {}, {}
        for code, time, data in event_file:
            if code == 0 and not code_to_name:
                code_to_name = dict((c, data[c]['tagname']) for c in data)
                name_to_code = dict((data[c]['tagname'], c) for c in data)
                break
        codes = [name_to_code[name] for name in names ]
        for code, time, data in event_file:
            if code in codes and time > time_range[0] and time < time_range[1]:
                data_dict['code'].append(code)
                data_dict['name'].append(code_to_name[code])
                data_dict['time'].append(time)
                data_dict['data'].append(data)
    data_df = pd.DataFrame(data_dict)
    data_df = data_df.sort_values(by='time').reset_index(drop=True)
    return data_df



def get_events_mac(event_file, name):
    data = {
        'code': [],
        'name': [],
        'time': [],
        'data': [],
    }
    for event in event_file.get_events_iter(codes=name):
        data['code'].append(event.code)
        data['name'].append(event_file.codec[event.code])
        data['time'].append(event.time)
        data['data'].append(event.data)
    data = pd.DataFrame(data)
    data = data.sort_values(by='time').reset_index(drop=True)
    return data


def dump_events(filename, photodiode_file, sample_on_file, artefact_diode_flag = False, reverse_flag=False, skip_num=None):
    print(filename)
    
    if args.date:
        print('Date exists')
        baseDir = os.path.join('/om/user/ssazaidi/braintree_data/projects/', config['Experiment Information']['name'], 'monkeys', config['Experiment Information']['monkey'])
        mworksRawDir = os.path.join(baseDir, 'mworksraw')
        intanRawDir = os.path.join(baseDir, 'intanraw')
        rawDataDir = intanRawDir
        mworksprocDir = os.path.join(baseDir, 'mworksproc')
        intanRawDir = os.path.join(intanRawDir, [i for i in os.listdir(intanRawDir) if date+'_' in i][args.session_num])

        filename = os.path.join(mworksRawDir,[i for i in os.listdir(mworksRawDir) if date in i][args.session_num])

        photodiode_file = os.path.join(intanRawDir, 'board-ANALOG-IN-1.dat')
        sample_on_file = os.path.join(intanRawDir, 'board-DIGITAL-IN-02.dat')

    else:
        print('date does not exist: ', args.date)
        if filename == '':
            print('Specify either filenames or the date')
            return
        rawDataDir = '/'.join(sample_on_file.split('/')[:-2])
        mworksprocDir = '/'.join(sample_on_file.split('/')[:-3] + ['mworksproc'])

    
    print(rawDataDir)
    print(filename)
    print(photodiode_file)
    print(sample_on_file)
    print(args.remove_artefacts)
    
#     return
    
    if (not os.path.exists(rawDataDir) )and artefact_diode_flag:
        print('Artefact times not available')
        return -1
#     event_file = MWKFile(filename)
#     event_file.open()
    print(skip_num)
    if skip_num:
        skip_num = int(skip_num)
    # return  
    # Variables we'd like to fetch data for
    names = ['trial_start_line',
             'correct_fixation',
             'stimulus_presented',
             'stim_on_time',
             'stim_off_time',
             'stim_on_delay',
             'stimulus_size',
             'fixation_window_size',
             'fixation_point_size_min']
    
    if 'stimulation' in config['Experiment Information']['name']:
        names = names+['stim_key', 'stim_id',  'stim_current',]
    
    data_file_name = os.path.join(rawDataDir,date, 'all_data.pkl')
    
    if os.path.exists(data_file_name):
        data = joblib.load(data_file_name)
    else:
        data = get_events(filepath=filename, names=names)
        os.makedirs(os.path.join(rawDataDir,date), exist_ok = True)
        joblib.dump(data, data_file_name)
    # event_file.close()

    ###########################################################################
    # Create a dict to store output information
    ###########################################################################
    output = {
        'stim_on_time_ms': data[data.name == 'stim_on_time']['data'].values[-1] / 1000.,
        'stim_off_time_ms': data[data.name == 'stim_off_time']['data'].values[-1] / 1000.,
        'stim_on_delay_ms': data[data.name == 'stim_on_delay']['data'].values[-1] / 1000.,
        'stimulus_size_degrees': data[data.name == 'stimulus_size']['data'].values[-1],
        'fixation_window_size_degrees': data[data.name == 'fixation_window_size']['data'].values[-1],
        'fixation_point_size_degrees': data[data.name == 'fixation_point_size_min']['data'].values[-1],
    }

    ###########################################################################
    # Add column in data to indicate whether stimulus was first in trial or not
    ###########################################################################
    data['first_in_trial'] = False
    # Filter data to only get `trial_start_line` and `stimulus_presented` information
    df = data[(data.name == 'trial_start_line') | ((data.name == 'stimulus_presented') & (data.data != -1))]
    # Extract `time` for the first `stimulus_presented` (which is right after `trial_start_line` has been pulsed)
    first_in_trial_times = [df.time.values[i] for i in range(1, len(df))
                            if ((df.name.values[i - 1] == 'trial_start_line') and
                                (df.name.values[i] == 'stimulus_presented'))]
    data['first_in_trial'] = data['time'].apply(lambda x: True if x in first_in_trial_times else False)

    ###########################################################################
    # Extract stimulus presentation order and fixation information
    ###########################################################################
    stimulus_presented_df = data[data.name == 'stimulus_presented'].reset_index(drop=True)
    correct_fixation_df = data[data.name == 'correct_fixation'].reset_index(drop=True)
    stimulus_presented_df = stimulus_presented_df[:len(correct_fixation_df)]  # If you have one extra stimulus event but not fixation, use this
    print(len(stimulus_presented_df) ,   len(correct_fixation_df))
    assert len(stimulus_presented_df) == len(correct_fixation_df)
    # Drop `empty` data (i.e. -1) before the experiment actually began and after it had already ended
    correct_fixation_df = correct_fixation_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    stimulus_presented_df = stimulus_presented_df[stimulus_presented_df.data != -1].reset_index(drop=True)
    # Add `first_in_trial` info to other data frame too
    correct_fixation_df['first_in_trial'] = stimulus_presented_df['first_in_trial']

    ###########################################################################
    # Add column to indicate order in trial (1 2 3 1 2 3 etc.)
    ###########################################################################
    assert stimulus_presented_df.iloc[0].first_in_trial
    stimulus_presented_df['stimulus_order_in_trial'] = ''
    counter = 1
    for index, row in stimulus_presented_df.iterrows():
        if row['first_in_trial']:
            counter = 1
        stimulus_presented_df.at[index, 'stimulus_order_in_trial'] = counter
        counter += 1
    correct_fixation_df['stimulus_order_in_trial'] = stimulus_presented_df['stimulus_order_in_trial']

    ###########################################################################
    # Read sample on file
    ###########################################################################
    fid = open(sample_on_file, 'r')
    filesize = os.path.getsize(filename)  # in bytes
    num_samples = filesize // 2  # uint16 = 2 bytes
    digital_in = np.fromfile(fid, 'uint16', num_samples)
    fid.close()

    samp_on, = np.nonzero(digital_in[:-1] < digital_in[1:])  # Look for 0->1 transitions
    samp_on = samp_on + 1  # Previous line returns indexes of 0s seen before spikes, but we want indexes of first spikes
    len_samp_on = len(samp_on)
    len_stimulus_presented = len(stimulus_presented_df)
    
    if len(stimulus_presented_df) > len_samp_on:
        print('Warning: Trimming MWorks files as {} > {}'.format(len(stimulus_presented_df),len(samp_on)))
        if reverse_flag:
            if skip_num:
                start_sample = - len_samp_on - skip_num
                end_sample = - skip_num
            else:
                start_sample = -len_samp_on
                end_sample = skip_num
            stimulus_presented_df = stimulus_presented_df[start_sample:end_sample]
            correct_fixation_df = correct_fixation_df[start_sample:end_sample]
        else:
            stimulus_presented_df = stimulus_presented_df[skip_num:len_samp_on]
            correct_fixation_df = correct_fixation_df[skip_num:len_samp_on]

    samp_on = samp_on[:len(correct_fixation_df)]   # If you have one extra stimulus event but not fixation, use this
    print(len_samp_on,  len(stimulus_presented_df))
    assert len(samp_on) == len(stimulus_presented_df)

    ###########################################################################
    # Read photodiode file
    ###########################################################################
    if artefact_diode_flag:
        print(filename)
        
        artefact_times = joblib.load(os.path.join(rawDataDir, date, 'artefact_time_'+date+'.pkl'))
        nan_sums = np.sum(np.isnan(artefact_times), axis=0)
        good_ones = np.where(nan_sums < 96)[0]
        
        artefact_times = np.nanmedian(artefact_times, axis=0).astype(int)
        
        photodiode_on = artefact_times
        print(' length of photo_diode_signal', len(photodiode_on))
    else:    
        fid = open(photodiode_file, 'r')
        filesize = os.path.getsize(photodiode_file)  # in bytes
        num_samples = filesize // 2  # uint16 = 2 bytes
        v = np.fromfile(fid, 'uint16', num_samples)
        fid.close()

        # Convert to volts (use this if the data file was generated by Recording Controller)
        v = (v - 32768) * 0.0003125

        # Detect rises in the oscillating photodiode signal
        peaks, _ = find_peaks(v, height=0)  # Find all peaks
        peaks = np.asarray([p for p in peaks if v[p] > THRESHOLD])  # Apply threshold
        photodiode_on = np.asarray([min(peaks[(peaks >= s) & (peaks < (s + 100000))]) for s in samp_on])

        assert len(photodiode_on) == len(stimulus_presented_df)

        # Convert both times to microseconds to match MWorks
    photodiode_on = photodiode_on * 1000000 / SAMPLING_FREQUENCY_HZ  # in us
    samp_on = samp_on * 1000000 / SAMPLING_FREQUENCY_HZ  # in us

    ###########################################################################
    # Correct the times
    ###########################################################################
    corrected_time = photodiode_on  # Both are in microseconds
    print(f'Delay recorded on photodiode is {np.mean(np.abs(photodiode_on - samp_on)) / 1000.:.2f} ms on average')

    stimulus_presented_df['time'] = corrected_time
    correct_fixation_df['time'] = corrected_time

    # Print any times differences between digital signal and photodiode that are atrociously huge (>40ms)
    for i, x in enumerate(photodiode_on - samp_on):
        if x / 1000. > 40 or x/1000 < -40:
            print(f'Warning: Sample {i} has delay of {x / 1000.} ms')

    ###########################################################################
    # Get eye data
    ###########################################################################
#     eye_h, eye_v, eye_time = [], [], []
#     pupil_size, pupil_time = [], []
#     for t in stimulus_presented_df.time.values:
#         t1 = int(t - 50 * 1000.)  # Start time (ms)
#         t2 = int(t + (output['stim_on_time_ms'] + 50) * 1000.)  # Stop time (ms)
#         h = [event.data for event in get_events(filename, ['eye_h'], time_range=[t1, t2])]
#         v = [event.data for event in get_events(filename, ['eye_v'], time_range=[t1, t2])]
#         time = [(event.time - t) / 1000. for event in get_events(codes=['eye_v'], time_range=[t1, t2])]
#         assert len(h) == len(v)
#         assert len(time) == len(h)
#         eye_h.append(h)
#         eye_v.append(v)
#         eye_time.append(time)
        # t1 = int(t - 1000 * 1000.)  # Start time (ms)
        # t2 = int(t + (output['stim_on_time_ms'] + 2000) * 1000.)  # Stop time (ms)
        # p = [event.data for event in event_file.get_events_iter(codes=['pupil_size_r'], time_range=[t1, t2])]
        # p_time = [(event.time - t) / 1000. for event in event_file.get_events_iter(codes=['pupil_size_r'], time_range=[t1, t2])]
        # assert len(p_time) == len(p)
        # pupil_size.append(p)
        # pupil_time.append(p_time)
#     assert len(eye_h) == len(stimulus_presented_df)
    # assert len(pupil_size) == len(stimulus_presented_df)
#     event_file.close()

    ###########################################################################
    # Double-check `correct_fixation` is actually correct by analyzing the
    # `eye_h` and `eye_v` data
    ###########################################################################
    # # Threshold to check against to determine if we have enough eye data for given stimulus presentation
    # threshold = output['stim_on_time_ms'] // 2
    #
    # for i in range(len(eye_h)):
    #     if correct_fixation_df.iloc[i]['data'] == 0:  # Skip if already marked incorrect
    #         continue
    #
    #     if len(eye_h[i]) < threshold or len(eye_v[i]) < threshold:
    #         correct_fixation_df.at[i, 'data'] = 0
    #     elif np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_h[i]]) or\
    #             np.any([np.abs(_) > output['fixation_window_size_degrees'] for _ in eye_v[i]]):
    #         correct_fixation_df.at[i, 'data'] = 0

    ################################################################################
    # Prepare Current and ID arrays
    ################################################################################

    if artefact_diode_flag:
    
        current_events = data.loc[data['name'] == 'stim_current'].reset_index(drop=True)
        id_events = data.loc[data['name'] == 'stim_id'].reset_index(drop=True)
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

        output['stim_id'] = np.array(id_events.iloc[correct_id].data)[:len(samp_on)][good_ones]

        output['stim_current'] = np.array(current_events.iloc[correct_current].data)[:len(samp_on)][good_ones]
    else:
        good_ones = np.arange(len(samp_on))
        
    ###########################################################################
    # Save output
    ###########################################################################
    stimulus_presented = stimulus_presented_df.data.values[good_ones].tolist()
    samp_on_id = output['stim_id']
    samp_on_current = output['stim_current']
    temp = np.unique(list(zip(stimulus_presented,samp_on_id,samp_on_current)), axis=0)
    stim_num_key = dict(zip([tuple( i )for i in temp], np.arange(1,len(temp)+1)))
    new_stim_presented = [stim_num_key[tuple(i)] for i in np.array(list(zip(stimulus_presented,samp_on_id,samp_on_current)))]
    output['stimulus_presented'] = new_stim_presented
    output['fixation_correct'] = correct_fixation_df.data.values[good_ones].tolist()
    output['stimulus_order_in_trial'] = stimulus_presented_df.stimulus_order_in_trial.values[good_ones].tolist()
#     output['eye_h_degrees'] = eye_h
#     output['eye_v_degrees'] = eye_v
#     output['eye_time_ms'] = eye_time
    output['samp_on_us'] = samp_on.astype(int)[good_ones]  # Convert to int okay only if times are in microseconds
    output['photodiode_on_us'] = photodiode_on.astype(int)[good_ones]  # Convert to int okay only if times are in microseconds
    # output['pupil_size_degrees'] = pupil_size
    # output['pupil_time_ms'] = pupil_time

    output = pd.DataFrame(output)
    os.makedirs(mworksprocDir, exist_ok=True)
    output.to_csv(os.path.join(mworksprocDir, filename.split('/')[-1][:-5] + '_mwk.csv'), index=False)  # -5 in filename to delete the .mwk2 extension

    ###########################################################################
    # Repetitions
    ###########################################################################
    selected_indexes = np.array(correct_fixation_df[correct_fixation_df.data == 1]['data'].index.tolist())
    # if reverse_flag:
    #     selected_indexes = selected_indexes - len_samp_on - skip_num
    # elif skip_num:
    #     selected_indexes = selected_indexes - skip_num

    selected_indexes = selected_indexes - np.min(selected_indexes)

    correct_trials = np.asarray(stimulus_presented_df.data.values.tolist())[selected_indexes]
    num_repetitions = np.asarray([len(correct_trials[correct_trials == stimulus]) for stimulus in np.unique(stimulus_presented_df.data.values.tolist())])
    print(f'{min(num_repetitions)} repeats, range is {np.unique(num_repetitions)}')


if __name__ == '__main__':
    dump_events(args.mworks_file,args.photodiode_file,args.samp_on_file, args.remove_artefacts)
    # dump_events(sys.argv[1], sys.argv[2], sys.argv[3])

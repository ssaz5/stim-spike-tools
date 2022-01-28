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
parser.add_argument('mworks_file', metavar='N', nargs='?', type=str,
    help='Mworks file Name: To be changed soon')
parser.add_argument('--session_num', type=int, nargs='?',default=0)
parser.add_argument('--date', type=str)
parser.add_argument('--project_name', type=str)

args = parser.parse_args()


date = args.date
if not date:
    date = config['Experiment Information']['date']


project_name = args.project_name
if project_name:
    config['Experiment Information']['name'] = project_name


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


def dump_events(filename):
    if args.date:
        print('Date exists')
        baseDir = os.path.join('/om/user/ssazaidi/braintree_data/projects/', config['Experiment Information']['name'], 'monkeys', config['Experiment Information']['monkey'])
        mworksRawDir = os.path.join(baseDir, 'mworksraw')
        intanRawDir = os.path.join(baseDir, 'intanraw')
        rawDataDir = intanRawDir
        mworksprocDir = os.path.join(baseDir, 'mworksproc')
        intanRawDir = os.path.join(intanRawDir, [i for i in os.listdir(intanRawDir) if date+'_' in i][args.session_num])

        filename = os.path.join(mworksRawDir,[i for i in os.listdir(mworksRawDir) if date in i][args.session_num])

    else:
        rawDataDir = '/'.join(filename.split('/')[:-2]+['intanraw'])
        mworksprocDir = '/'.join(filename.split('/')[:-2] + ['mworksproc'])
    data_dir_name = os.path.join(rawDataDir,date)
    data_file_name = os.path.join(data_dir_name, 'all_data_'+str(args.session_num)+'.pkl')
    names = ['trial_start_line',
             'stim_key',
             'stim_id',
             'stim_current',
             'correct_fixation',
             'stimulus_presented',
             'stim_on_time',
             'stim_off_time',
             'stim_on_delay',
             'stimulus_size',
             'fixation_window_size',
             'fixation_point_size_min']
    print(rawDataDir, mworksprocDir, data_file_name)
#     return
    if os.path.exists(data_file_name):
        data = joblib.load(data_file_name)
    else:
        print('Making Directory')
        os.makedirs(data_dir_name, exist_ok = True)
        data = get_events(filepath=filename, names=names)
        joblib.dump(data, data_file_name)



if __name__ == '__main__':
    dump_events(args.mworks_file)
    # dump_events(sys.argv[1], sys.argv[2], sys.argv[3])

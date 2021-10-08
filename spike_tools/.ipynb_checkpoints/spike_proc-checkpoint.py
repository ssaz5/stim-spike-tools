import sys
from pathlib import Path
import configparser
import argparse
import logging

from utils.spikeutils import get_spike_times, get_psth, combine_channels, combine_sessions

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(Path(__file__).parent / 'spike_config.ini')

parser = argparse.ArgumentParser(description='Run spike detection.')
parser.add_argument('num', metavar='N', type=int,
    help='channel number or slurm job array id')
parser.add_argument('--flow', type=int)
parser.add_argument('--fhigh', type=int)
parser.add_argument('--fsampling', type=int)
parser.add_argument('--threshold', type=int)
parser.add_argument('--date', type=str)
parser.add_argument('--paradigm', type=str)
parser.add_argument('--monkey', type=str)
parser.add_argument('--nchannels', type=int)
parser.add_argument('--starttime', type=int)
parser.add_argument('--stoptime', type=int)
parser.add_argument('--timebin', type=int)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('-n',  '--num_images', type=int)
parser.add_argument('-no',  '--normalize', type=int, default=0)
parser.add_argument('-np',  '--num_pulses', type=int, default=10)
parser.add_argument('-pp',  '--pulse_period', type=int, default=4000)
parser.add_argument('-e', '--experiment_name', type=str, default='spike_times')
parser.add_argument('-o', '--output_dir', type=str)
parser.add_argument('-ds', '--dates', type=str, nargs='+')
parser.add_argument('-ra',  '--remove_stim_artefacts', type=int, default=0)
parser.add_argument('-sa',  '--apply_salpa', type=int, default=0)


args = parser.parse_args()

f_low = args.flow
if not f_low:
    f_low = config['Filtering'].getint('fLow')
f_high = args.fhigh
if not f_high:
    f_high = config['Filtering'].getint('fHigh')
f_sampling = args.fsampling
if not f_sampling:
    f_sampling = config['Filtering'].getint('fSampling')

noise_threshold = args.threshold
if not noise_threshold:
    noise_threshold = config['Threshold'].getfloat('noiseThreshold')

paradigm = args.paradigm
if not paradigm:
    paradigm = config['Experiment Information']['paradigm']
date = args.date
if not date:
    date = config['Experiment Information']['date']
monkey = args.monkey
if not monkey:
    monkey = config['Experiment Information']['monkey']
n_channels = args.nchannels
if not n_channels:
    n_channels = config['Experiment Information'].getint('n_channels')

start_time = args.starttime
if not start_time:
    start_time = config['PSTH'].getint('startTime')
stop_time = args.stoptime
if not stop_time:
    stop_time = config['PSTH'].getint('stopTime')
timebin = args.timebin
if not timebin:
    timebin = config['PSTH'].getint('timebin')
    
output_dir = args.output_dir
if not output_dir:
    output_dir = config['Paths']['proc_dir']
    
dates = args.dates
if not dates:
    dates = config['Experiment Information']['date']

raw_dir = config['Paths']['raw_dir']
proc_dir = config['Paths']['proc_dir']

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


if args.experiment_name == 'spike_times':
    get_spike_times(args.num, date, raw_dir, proc_dir, f_sampling, n_channels, f_low, f_high, noise_threshold, remove_stim_artefacts = args.remove_stim_artefacts, num_pulses= args.num_pulses, pulse_period= args.pulse_period, apply_salpa=args.apply_salpa)
elif args.experiment_name == 'psth': 
    get_psth(args.num, date, proc_dir, start_time, stop_time, timebin, args.num_images)
elif args.experiment_name == 'combine_channels':
    combine_channels(proc_dir, n_channels, args.suffix)
elif args.experiment_name == 'combine_sessions':
    combine_sessions(dates, proc_dir, output_dir, normalize=args.normalize, save_format='h5')
    print(dates)
else:
    pass

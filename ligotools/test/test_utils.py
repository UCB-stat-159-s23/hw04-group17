import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# LIGO-specific readligo.py 
from ligotools import readligo as rl

# import functions from ligotools.utils
from ligotools import utils
from scipy.io import wavfile

fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))

event = events[eventname]
fn_H1 = event['fn_H1']             
fn_L1 = event['fn_L1']              
fn_template = event['fn_template']  
fs = event['fs']                   
tevent = event['tevent']           
fband = event['fband']

strain_H1, time_H1, chan_dict_H1 = rl.loaddata("data/"+fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata("data/"+fn_L1, 'L1')

NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)
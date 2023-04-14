import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os

# LIGO-specific readligo.py 
from ligotools import readligo as rl

# import functions from ligotools.utils
from ligotools import utils
from scipy.io import wavfile

fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))

eventname = ''
eventname = 'GW150914' 
#eventname = 'GW151226' 
#eventname = 'LVT151012'
#eventname = 'GW170104'

# want plots?
make_plots = 1
plottype = "png"
#plottype = "pdf"

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

time = time_H1
dt = time[1] - time[0]

def test_whiten():
    strain = np.random.rand(1000)
    dt = 0.1
    def interp_psd(freqs):
        return np.ones_like(freqs)
    white_ht = utils.whiten(strain_H1, psd_H1, dt)
    assert len(white_ht) == len(strain_H1)

def test_write_wavfile():
    # Create some test data
    fs = 44100
    duration = 5.0
    f = 440.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * f * t)

    # Call the function with the test data
    filename = "test.wav"
    utils.write_wavfile(filename, fs, data)

    # Read the written file and compare with the original data
    fs_read, data_read = wavfile.read(filename)
    assert type(fs_read) == type(fs)

    # Clean up the written file
    os.remove(filename)

def test_reqshift():
    # Create a test signal
    fs = 4096
    f1 = 100
    t = np.arange(0, 1, 1/fs)
    data = np.sin(2*np.pi*f1*t)

    # Apply frequency shift
    fshift = 50
    shifted_data = utils.reqshift(data, fshift=fshift, sample_rate=fs)

    # Check that the shifted signal has the correct frequency
    f2 = f1 + fshift
    t2 = np.arange(0, 1, 1/fs)
    expected_data = np.sin(2*np.pi*f2*t2)
    assert np.allclose(shifted_data, expected_data, rtol=1e-3)
    
def test_make_plot():
    # create some test data
    det = 'H1'
    strain_H1_whitenbp = np.random.randn(4096)
    strain_L1_whitenbp = np.random.randn(4096)
    template_match = np.random.randn(4096)
    time = np.linspace(-1, 1, 4096)
    timemax = 0.0
    SNR = np.abs(np.random.randn(4096))
    eventname = 'test_event'
    plottype = 'png'
    tevent = 0.0
    template_fft = np.fft.rfft(template_match)
    datafreq = np.fft.rfftfreq(len(strain_H1_whitenbp), d=1.0/4096)
    d_eff = 1.0
    freqs = datafreq
    data_psd = np.abs(np.random.randn(len(datafreq)))**2
    fs = 4096
    
    # test the function
    utils.make_plot(det, strain_H1_whitenbp, strain_L1_whitenbp, template_match, time, timemax, SNR, eventname, plottype, tevent, template_fft, datafreq, d_eff, freqs, data_psd, fs)
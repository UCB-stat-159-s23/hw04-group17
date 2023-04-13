import numpy as np
import pytest
from ligotools import readligo as rl

file_L1 = "data/L-L1_LOSC_4_V2-1126259446-32.hdf5"
file_H1 = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"

strain_H1, time_H1, chan_dict_H1 = rl.loaddata(file_L1)
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(file_H1)
strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = rl.read_hdf5("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")

def test_loaddata_L1():
	assert isinstance(strain_L1, np.ndarray)
	assert isinstance(time_L1, np.ndarray)
	assert isinstance(chan_dict_L1, dict)
	
	
def test_loaddata_H1():
	assert isinstance(strain_H1, np.ndarray)
	assert isinstance(time_H1, np.ndarray)
	assert isinstance(chan_dict_H1, dict)
	
def test_read_hdf5():
	assert srain != 0
	
def test_getsegs():
	start = 1126259446
	stop = 'H1'
	ifo = 'H1'
	flag = 'DATA'
	segList = readligo.getsegs(start, stop, ifo, flag)
	assert segList.starttime == start
	assert segList.endtime == stop
	assert segList.ifo == ifo
	assert len(segList) == 2
	assert segList[0] == (1126259446, 1126259447)
	assert segList[1] == (1126259452, 1126259454)
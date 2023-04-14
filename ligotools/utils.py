from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import filtfilt

def whiten(strain, interp_psd, dt):
    """
    This function whitens a strain time series by dividing it by the interpolated power spectral density (PSD).
    The resulting time series has a flat frequency response, i.e., equal power at all frequencies.

    Parameters:
        strain (array): The strain time series to be whitened.
        interp_psd (function): A function that interpolates the PSD at arbitrary frequencies.
        dt (float): The time step of the strain time series.

    Returns:
        array: The whitened strain time series.
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename, fs, data):
    """
    Normalize data to ensure that it is within integer limits, and write to wavfile.

    Parameters:
    -----------
    filename : str
        The name of the file to write the data to.
    fs : int
        The sample rate of the data.
    data : numpy.ndarray
        The data to be written to the file.

    Returns:
    --------
    None
    """
    # Normalize the data to ensure that it is within integer limits
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)

    # Write the data to the specified file
    wavfile.write(filename, fs, d)




def reqshift(data, fshift=100, sample_rate=4096):
    """
    Shift the frequency of the signal by a constant amount.

    Parameters:
    -----------
    data : numpy.ndarray
        The signal to be shifted.
    fshift : float, optional
        The frequency shift to be applied (in Hz).
    sample_rate : int, optional
        The sample rate of the data (in Hz).

    Returns:
    --------
    numpy.ndarray
        The frequency-shifted signal.
    """
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.
    z = np.fft.irfft(y)
    return z


def plot_all(eventname, det, plottype, time, timemax, tevent, SNR, strain_whitenbp, template_match, pcolor, datafreq, template_fft, d_eff, freqs, data_psd, bb, ab, normalization):
    # -- Plot the result
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-timemax, SNR, pcolor, label=det+' SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.title(det+' matched filter SNR around event')

    # zoom in
    plt.subplot(2,1,2)
    plt.plot(time-timemax, SNR, pcolor, label=det+' SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.savefig(eventname+"_"+det+"_SNR."+plottype)

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-tevent, strain_whitenbp, pcolor, label=det+' whitened h(t)')
    plt.plot(time-tevent, template_match, 'k', label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' whitened data around event')

    plt.subplot(2,1,2)
    plt.plot(time-tevent, strain_whitenbp-template_match, pcolor, label=det+' resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' Residual whitened data after subtracting template around event')
    plt.savefig(eventname+"_"+det+"_matchtime."+plottype)
                 
    # -- Display PSD and template
    plt.figure(figsize=(10,6))
    template_f = np.absolute(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd), pcolor, label=det+' ASD')
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det+' ASD and template around event')
    plt.savefig(eventname+"_"+det+"_matchfreq."+plottype)

    
    # plot function
def make_plot(det,strain_H1_whitenbp,strain_L1_whitenbp,template_match,time,timemax,SNR,eventname,plottype,tevent,template_fft,datafreq,d_eff,freqs,data_psd,fs):

    # plotting changes for the detectors:
    if det == 'L1': 
        pcolor='g'
        strain_whitenbp = strain_L1_whitenbp
        template_L1 = template_match.copy()
    else:
        pcolor='r'
        strain_whitenbp = strain_H1_whitenbp
        template_H1 = template_match.copy()

    # -- Plot the result
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    #plt.ylim([0,25.])
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.title(det+' matched filter SNR around event')

    # zoom in
    plt.subplot(2,1,2)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlim([-0.15,0.05])
    #plt.xlim([-0.3,+0.3])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.savefig("figures/"+eventname+"_"+det+"_SNR."+plottype)

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
    plt.plot(time-tevent,template_match,'k',label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' whitened data around event')

    plt.subplot(2,1,2)
    plt.plot(time-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' Residual whitened data after subtracting template around event')
    plt.savefig("figures/"+eventname+"_"+det+"_matchtime."+plottype)

    # -- Display PSD and template
    # must multiply by sqrt(f) to plot template fft on top of ASD:
    plt.figure(figsize=(10,6))
    template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det+' ASD and template around event')
    plt.savefig("figures/"+eventname+"_"+det+"_matchfreq."+plottype)
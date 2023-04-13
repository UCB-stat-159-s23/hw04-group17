from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt


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


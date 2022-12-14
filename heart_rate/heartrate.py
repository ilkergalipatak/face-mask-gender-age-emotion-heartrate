from scipy import signal

def find_heart_rate(fft,freqs,freq_min,freq_max):
    fft_maximums=[]

    for i in range(fft.shape[0]):
        if freq_min<=freqs[i]<=freq_max:
            fft_map=abs(fft[i])
            fft_maximums.append(fft_map.max())
        else:
            fft_maximums.append(0)
    peaks, properties=signal.find_peaks(fft_maximums)
    max_peak=-1
    max_freq=0

    for peak in peaks:
        if fft_maximums[peak]>max_freq:
            max_freq=fft_maximums[peak]
            max_peak=peak
    return freqs[max_peak]*60
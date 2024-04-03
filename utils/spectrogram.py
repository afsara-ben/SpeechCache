import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram

wav_path = '/Users/afsarabenazir/Downloads/speech_datasets/slurp/audio/slurp_real/'
file1 = wav_path + 'audio-1497884011.flac'
file2 = wav_path + 'audio-1497884648.flac'
file3 = wav_path + 'audio-1497884648-headset.flac'
# Audio file paths
audio_files = [
file1, file2, file3
]

# waveform, sample_rate = sf.read(file2)
#
# # Define the time range to zoom in on (in seconds)
# zoom_start_time = 0.05
# zoom_end_time = 0.1
#
# # Calculate the corresponding sample indices
# zoom_start_index = int(zoom_start_time * sample_rate)
# zoom_end_index = int(zoom_end_time * sample_rate)
#
# # Extract the zoomed-in portion of the waveform
# zoomed_waveform = waveform[zoom_start_index:zoom_end_index]
#
# # Calculate the time axis for the zoomed-in portion
# zoomed_time = np.linspace(zoom_start_time, zoom_end_time, num=len(zoomed_waveform))
#
# # Plot the zoomed-in waveform
# plt.figure(figsize=(10, 4))
# plt.plot(zoomed_time, zoomed_waveform, color='b')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Zoomed-In Waveform')
# plt.grid(True)
# plt.show()
#
# Create subplots for waveforms and spectrograms
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16), gridspec_kw={'hspace': 0.6, 'wspace': 0.3})

# Process each audio file
for i, audio_file in enumerate(audio_files):
    # Load the speech waveform from the audio file
    waveform, sample_rate = sf.read(audio_file)

    # Calculate the time axis in seconds
    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, num=len(waveform))

    # Plot the speech waveform
    axes[i, 0].plot(time, waveform, color='royalblue')
    axes[i, 0].set_xlabel('Time (s)')
    axes[i, 0].set_ylabel('Amplitude')
    axes[i, 0].set_title('Speech Waveform')
    axes[i, 0].grid(True)

    # Calculate and plot the spectrogram
    frequencies, times, Sxx = spectrogram(waveform, sample_rate)
    axes[i, 1].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='jet')
    axes[i, 1].set_xlabel('Time (s)')
    axes[i, 1].set_ylabel('Frequency (Hz)')
    axes[i, 1].set_title('Spectrogram')
    fig.colorbar(axes[i, 1].collections[0], ax=axes[i, 1], label='Power Spectral Density (dB)')

# Adjust the layout and spacing
plt.tight_layout()

# Set the output image size
plt.gcf().set_size_inches(8, 10)
# Display the plot
plt.show()
plt.savefig('/Users/afsarabenazir/Downloads/figs/spectrogram.png')

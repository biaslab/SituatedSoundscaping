using WAV
using DSP
using BenchmarkTools
using Statistics
using GaussianMixtures
using PyPlot

include("src/io.jl")
include("src/utils.jl")
include("src/processing.jl")
include("src/train.jl")
include("src/separation.jl")

# settings
audio_files = ["audio/woman.wav",               # paths to audio files
               "audio/cafeteria.wav"]      
fs = 16000                                      # desired sampling frequency [Hz]
subtract_mean = true;                           # subtract the mean during preprocessing
normalize_std = true;                           # normalize with standard deviation during preprocessing
power_levels = [0, -5]                          # specify desired signal amplification in the mixture [dB]
duration_train = 3                              # duration of training data [sec]
duration_test = 10                              # duration of testing data [sec]
offset = 1                                      # offset in audio signal [sec]
block_length = 100                              # length of processing blocks [samples]
block_overlap_train = 90                        # overlap inbetween blocks during training [samples]
block_overlap_test = 50                         # overlap inbetween blocks during testing [samples]
nr_clusters = [15, 10]                          # number of clusters per model
σ2_noise = 1e-4                                 # observation noise variance

# load and preprocess data 
x_train, x_test = load_data(audio_files; fs=fs, duration=[duration_train, duration_test], offset=offset, levels_dB=power_levels, subtract_mean=subtract_mean, normalize_std=normalize_std);
x_mixture = sum(x_test);

# calculate spectra of signals
X_train, logX_train = calculate_spectra_broadcast(x_train, block_length, block_overlap_train; onesided=true, fs=fs, window=hanning);
X_test, logX_test = calculate_spectra_broadcast(x_test, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning);
X_mixture, logX_mixture = calculate_spectra(x_mixture, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning);

# train models 
models = train_GSMM.(X_train, logX_train, nr_clusters; its=100)

# perform informed source separation
x_separated, X_sep = separate_sources(X_mixture, models, σ2_noise)

## pseudo code
# [✓] load data
# [✓] convert into freq + log-power fragments
# [✓] train
# [✓] test

# [ ] smart cluster assignment

@btime separate_sources(X_mixture, models, σ2_noise)


plt.clf() 
_ , ax = plt.subplots(nrows=3, figsize=(15,15))
ax[1].plot(x_mixture)
ax[2].plot(x_test[1].-5)
ax[2].plot(x_separated[1].+5)
ax[3].plot(x_test[2].-5)
ax[3].plot(x_separated[2].+5)
plt.gcf()

plt.clf()
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
ax[1].imshow(models[1].μ, origin="lower")
ax[2].imshow(models[2].μ, origin="lower")
plt.gcf()

plt.clf()
plt.imshow(log.(abs2.(collect(transpose(X_sep[2])))), origin="lower", aspect="auto")
plt.colorbar()
#plt.clim(8,-17)
plt.gcf()

plt.clf()
plt.imshow(log.(abs2.(collect(transpose(X_test[1])))), origin="lower")
plt.colorbar()
plt.gcf()


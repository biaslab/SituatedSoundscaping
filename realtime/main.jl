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

# settings
audio_files = ["audio/woman.wav",               # paths to audio files
               "audio/cocktail_party.wav"]      
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

# load and preprocess data 
x_train, x_test = load_data(audio_files; fs=fs, duration=[duration_train, duration_test], offset=offset, levels_dB=power_levels, subtract_mean=subtract_mean, normalize_std=normalize_std);
x_mixture = sum(x_test);

# calculate spectra of signals
X_train, logX_train = calculate_spectra_broadcast(x_train, block_length, block_overlap_train; onesided=true, fs=fs, window=hanning);
X_test, logX_test = calculate_spectra_broadcast(x_test, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning);
X_mixture, logX_mixture = calculate_spectra(x_mixture, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning);

# train models 
models = train_GSMM.(X_train, logX_train, nr_clusters; its=10)

# perform informed source separation
x_sep = separate_sources(x_mixture, models)



## pseudo code
# [✓] load data
# [✓] convert into freq + log-power fragments
# [✓] train
# [ ] test
# [ ] smart cluster assignment

logX2 = logX_train[1]
X = X_train[1]
# stage 1 & 2: K-means & EM training
gmm = GMM(25, logX2, nIter=50, nInit=100, kind=:diag)
em!(gmm, logX2)

plt.clf()
plt.imshow(transpose(logX_train[1]), aspect="auto", origin="lower")
plt.colorbar()
plt.clim(-15,5)
gcf()
plt.imshow(params[2].μ, origin="lower")
gcf()

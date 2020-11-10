using WAV
using DSP
using BenchmarkTools
using Statistics, Dates
using GaussianMixtures
using PyPlot

include("src/io.jl")
include("src/utils.jl")
include("src/processing.jl")
include("src/train.jl")
include("src/separation.jl")
include("src/export.jl")

# settings
audio_files = ["audio/woman.wav",               # paths to audio files
               "audio/cocktailparty.wav"]      
fs = 16000                                      # desired sampling frequency [Hz]
subtract_mean = true;                           # subtract the mean during preprocessing
normalize_std = true;                           # normalize with standard deviation during preprocessing
power_levels = [0, 5]                           # specify desired signal amplification in the mixture [dB]
duration_train = 3                              # duration of training data [sec]
duration_test = 10                              # duration of testing data [sec]
offset = 1                                      # offset in audio signal [sec]
block_length = 160                              # length of processing blocks [samples]
block_overlap_train = 144                       # overlap inbetween blocks during training [samples]
block_overlap_test = 80                         # overlap inbetween blocks during testing [samples]
nr_clusters = [15, 15]                          # number of clusters per model
σ2_noise = 1e-5                                 # observation noise variance

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

# export results
export_results(x_separated, x_test, x_mixture, X_sep, X_test, X_mixture, power_levels, audio_files; fs=fs)

## pseudo code
# [✓] load data
# [✓] convert into freq + log-power fragments
# [✓] train
# [✓] test
# [✓] visualisation
# [ ] smart cluster assignment (extra)
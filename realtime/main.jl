using WAV
using DSP
using BenchmarkTools
using Statistics

include("src/io.jl")
include("src/utils.jl")
include("src/processing.jl")

# settings
audio_files = ["audio/woman.wav",               # paths to audio files
               "audio/cocktail_party.wav"]      
fs = 16000                                      # desired sampling frequency [Hz]
subtract_mean = true;                           # subtract the mean during preprocessing
normalize_std = true;                           # normalize with standard deviation during preprocessing
power_levels = [0, -5]                          # specify desired signal amplification in the mixture [dB]
duration_train = 3                              # duration of training data [sec]
duration_test = 10                              # duration of testing data [sec]
block_length = 100                              # length of processing blocks [samples]
block_overlap_train = 90                        # overlap inbetween blocks during training [samples]
block_overlap_test = 50                         # overlap inbetween blocks during testing [samples]


# load and preprocess data 
x_train, x_test = load_data(audio_files; fs=fs, duration=[duration_train, duration_test], levels_dB=power_levels, subtract_mean=subtract_mean, normalize_std=normalize_std)
x_mixture = sum(x_test)

# calculate spectra of signals
X_train, logX_train =calculate_spectra_broadcast(x_train, block_length, block_overlap_train; onesided=true, fs=fs, window=hanning)
X_test, logX_test = calculate_spectra_broadcast(x_test, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning)
X_mixture, logX_mixture = calculate_spectra(x_mixture, block_length, block_overlap_test; onesided=true, fs=fs, window=hanning)

# train models 
params = train_GSMM.()


## pseudo code
# [✓] load data
# [✓] convert into freq + log-power fragments
# [ ] train (smart cluster assignment???)
using WAV
using DSP
using BenchmarkTools
using Statistics

include("src/utils.jl")
include("src/processing.jl")

# settings
audio_files = ["audio/woman.wav",
               "audio/cocktail_party.wav"]
power_levels = [0,
                -5]

# load data
y, fs_tmp = wavread(audio_files[1])

fs = convert(Float64, fs_tmp)

y = squeeze(y)


# @code_warntype squeeze(y)
# @btime squeeze(y)
# @code_warntype squeeze!(y)
# @btime squeeze!(y)


# @code_warntype subtractMean!(y)
# @btime subtractMean!(y)

# @code_warntype normalizeVar!(y)
# @btime normalizeVar!(y)

# @code_warntype normalizeStd!(y)
# @btime normalizeStd!(y)


# var(y)

# addPGaindB!(y,10)

@btime preprocess(y, fs; fs=16000, duration=1, level_dB=0, subtract_mean=true, normalize_std=true)
@code_warntype preprocess(y, fs; fs=16000, duration=1, level_dB=0, subtract_mean=true, normalize_std=true)

@code_warntype load_data(audio_files; fs=16000, duration=1, levels_dB=power_levels, subtract_mean=true, normalize_std=true)
@btime load_data(audio_files; fs=16000, duration=1, levels_dB=power_levels, subtract_mean=true, normalize_std=true)

@code_warntype load_data2(audio_files; fs=16000, duration=1, levels_dB=power_levels, subtract_mean=true, normalize_std=true)
@btime load_data2(audio_files; fs=16000, duration=1, levels_dB=power_levels, subtract_mean=true, normalize_std=true)
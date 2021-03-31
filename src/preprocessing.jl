using WAV: wavread
using DSP: resample, stft
using Statistics: mean, std

export fetch_data

"""
This function loads the data from the specified files, preprocesses them and converts them to the desired domains.
"""
function fetch_data(settings::settings_structure)

    # give feedback to user
    @info "Fetching data..."

    # load time domain signals
    x = load_data_from_file(settings)

    # add converted domains
    y = convert_data(x, settings)

    # give feedback to user
    @info "The data has been fetched."

    # return signals
    return y

end


"""
This function converts time domain signals to the other domains.
"""
function convert_data(x::Array{Dict,1}, settings::settings_structure)

    # loop through signals
    for k_signal = 1:settings.nr_signals+1

        # calculate the short-time frequency spectrum
        X_train = stft(x[k_signal]["time_train"], settings.block_length, settings.block_overlap; onesided=settings.onesided, fs=settings.fs, window=settings.window)
        X_sep = stft(x[k_signal]["time_separation"], settings.block_length, settings.block_overlap; onesided=settings.onesided, fs=settings.fs, window=settings.window)

        # remove DC coefficient if desired
        if settings.include_DC 
            X_train = collect(transpose(X_train))
            X_sep = collect(transpose(X_sep))
        else
            X_train = collect(transpose(X_train[2:end,:]))
            X_sep = collect(transpose(X_sep[2:end,:]))
        end

        # remove DC coefficient if desired
        if settings.include_fs2 
            
        else
            X_train = X_train[:,1:end-1]
            X_sep = X_sep[:,1:end-1]
        end

        # calculate log-power spectrum 
        logX2_train = @. log(abs2(X_train))
        logX2_sep = @. log(abs2(X_sep))

        # save converted signals
        x[k_signal]["frequency_train"] = X_train
        x[k_signal]["logpower_train"] = logX2_train
        x[k_signal]["frequency_separation"] = X_sep
        x[k_signal]["logpower_separation"] = logX2_sep

    end
    
    # return spectra
    return x

end


"""
This function loads the data from the specified files and preprocesses it.
"""
function load_data_from_file(settings::settings_structure)

    # create array for signals
    y = Array{Dict,1}(undef, settings.nr_signals + 1)

    # loop through signals
    for k_signal = 1:settings.nr_signals

        # load file (first load sampling frequency to prevent loading all data)
        _, fs_tmp = wavread(settings.audio_files[k_signal]; subrange=1)
        fs_tmp = convert(Float64, fs_tmp)
        y_tmp_train, _ = wavread(settings.audio_files[k_signal]; subrange=fs_tmp*settings.offset_modeling:fs_tmp*(settings.duration_modeling+settings.offset_modeling))
        y_tmp_sep, _ = wavread(settings.audio_files[k_signal]; subrange=fs_tmp*(settings.duration_modeling+settings.offset_modeling):fs_tmp*(settings.duration_modeling+settings.offset_modeling+settings.duration_separation))
        y_tmp_train = y_tmp_train[:,1]
        y_tmp_sep = y_tmp_sep[:,1]
        y_tmp_train = squeeze(y_tmp_train)
        y_tmp_sep = squeeze(y_tmp_sep)

        # preprocess data
        y[k_signal] = Dict{String, Any}("time_train" => preprocess(y_tmp_train, fs_tmp, k_signal; settings, stage="train"),
                                        "time_separation" => preprocess(y_tmp_sep, fs_tmp, k_signal; settings, stage="separation"))
       
    end

    # add mixture signal to array
    y[end] = Dict{String, Any}("time_train"=> sum([y[k]["time_train"] for k=1:settings.nr_signals]),
                                "time_separation"=> sum([y[k]["time_separation"] for k=1:settings.nr_signals]),)

    # return signals
    return y

end


"""
Preprocesses signals
"""
function preprocess(x::Array{Float64,1}, fs_in::Float64, ind::Int64; settings::settings_structure, stage::String)

    # resample signal
    y = resample(x, settings.fs/fs_in)

    # crop signal
    if stage == "train"
        y = y[1:settings.duration_modeling*settings.fs]
    else
        y = y[1:Int(settings.duration_separation*settings.fs)]
    end

    # subtract mean
    if settings.subtract_mean
        subtractMean!(y)
    end

    # normalize power
    if settings.normalize_std
        normalizeStd!(y)
    end

    # add gain
    addPGaindB!(y, settings.power_levels[ind])

    # return preprocessed signal
    return y

end


"""
Adds a power gain to a signal in dB.
"""
function addPGaindB!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # correct gain factor
    gi = g/2
    gi = dB10toNum(gi)

    # amplify/suppress signal
    x .*= gi

end


"""
Adds a power gain to a signal.
"""
function addPGain!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # correct gain for power
    gi = sqrt(g)

    # amplify/suppress signal
    x .*= gi

end



"""
Adds an amplitude gain to a signal.
"""
function addAGain!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # amplify/suppress signal
    x .*= g

end



"""
Function that normalizes a signal with respect to its variance.
"""
function normalizeVar!(a::AbstractArray{T,N}) where {T,N}

    # calculate variance
    varvalue = var(a)

    # normalize by variance
     a ./= varvalue

end



"""
Function that normalizes a signal with respect to its standard deviation.
"""
function normalizeStd!(a::AbstractArray{T,N}) where {T,N}

    # calculate standard deviation
    stdvalue = std(a)

    # normalize by standard deviation
     a ./= stdvalue

end


"""
Function that subtracts the mean from an array.
"""
function subtractMean!(a::AbstractArray{T,N}) where {T,N}

    # calculate mean value
    meanvalue = mean(a)

    # subtract mean value
    a .-= meanvalue

end


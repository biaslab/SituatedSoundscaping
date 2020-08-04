# included functions:
#   calculate_spectra(x::AbstractArray{T,1}, block_length::Int64, block_overlap::Int64; onesided::Bool, fs::Int64, window::Function)
#   preprocess(y_tmp::AbstractArray{T,1}, fs_temp::Real; fs::Real, duration::Real, level_dB::Real, subtract_mean::Bool, normalize_std::Bool))
#   addPowerGaindB!(a::AbstractArray, g::Real)
#   addPowerGain!(a::AbstractArray, g::Real)
#   addAmplitudeGain!(a::AbstractArray, g::Real)
#   normalizeVar!(a::AbstractArray)
#   normalizeStd!(a::AbstractArray)
#   subtractMean!(a::AbstractArray)


#   calculate_spectra, calculate_spectra_broadcast
#
#   info:
#       This function calculates the frequency and log-power spectrum.
#
#   input arguments:
#       x::AbstractArray{T,1}           - Single dimension array with arbitrary contents
#       fs_in::Real                     - Input sampling rate
#       fs::Real                        - Desired sampling rate
#       duration::Real                  - Time of signal in seconds
#       level_dB::Real                  - Gain factor in dB
#       subtract_mean::Bool             - Flag whether to subtract the mean
#       normalize_std::Bool             - Flag whether to normalize its power to unity
#   output arguments:
#       y::Array{Float64,1}             - Array of preprocessed data

function calculate_spectra(x::AbstractArray{T,1}, block_length::Int64, block_overlap::Int64; onesided::Bool=true, fs::Int64=16000, window::Function=rect) where {T}

    # calculate the short-time frequency spectrum
    X = stft(x, block_length, block_overlap; onesided=onesided, fs=fs, window=window)

    # remove DC and fs/2
    X = X[2:end-1,:]

    # calculate log-power spectrum 
    logX2 = @. log(abs2(X))
    
    # return spectra
    return X, logX2

end

function calculate_spectra_broadcast(x::AbstractArray{T,1}, block_length::Int64, block_overlap::Int64; onesided::Bool=true, fs::Int64=16000, window::Function=rect) where {T}
    return  map(y->getindex.(calculate_spectra.(x, block_length, block_overlap; onesided=onesided, fs=fs, window=window), y), 1:2)
end


#   preprocess
#
#   info:
#       This function adds gain to a signal by multiplying the power with it.
#
#   input arguments:
#       x::AbstractArray{T,1}           - Single dimension array with arbitrary contents
#       fs_in::Real                     - Input sampling rate
#       fs::Real                        - Desired sampling rate
#       duration::Real                  - Time of signal in seconds
#       level_dB::Real                  - Gain factor in dB
#       subtract_mean::Bool             - Flag whether to subtract the mean
#       normalize_std::Bool             - Flag whether to normalize its power to unity
#   output arguments:
#       y::Array{Float64,1}             - Array of preprocessed data

function preprocess(x::AbstractArray{T,1}, fs_in::Real; fs::Real=16000, duration::Real=1, level_dB::Real=0, subtract_mean::Bool=true, normalize_std::Bool=true) where {T}

    # resample signal
    y = resample(x, fs_in/fs)

    # crop signal
    y = y[1:duration*fs]

    # subtract mean
    if subtract_mean
        subtractMean!(y)
    end

    # normalize power
    if normalize_std
        normalizeStd!(y)
    end

    # add gain
    addPGaindB!(y, level_dB)

    # return preprocessed signal
    return y

end


#   addPGaindB!
#
#   info:
#       This function adds gain to a signal by multiplying the power with it.
#
#   input arguments:
#       x::AbstractArray        - Array of arbitrary shape with arbitrary contents
#       g::Real                 - Gain factor in dB
#   output arguments:
#       _::AbstractArray        - Array of similar shape with amplified/suppressed contents

function addPGaindB!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # correct gain factor
    gi = g/2
    gi = dB10toNum(gi)

    # amplify/suppress signal
    x .*= gi

end


#   addAGain!
#
#   info:
#       This function adds gain to the power of the signal by multiplying the amplitude with its square root.
#
#   input arguments:
#       x::AbstractArray        - Array of arbitrary shape with arbitrary contents
#       g::Real                 - Gain factor
#   output arguments:
#       _::AbstractArray        - Array of similar shape with amplified/suppressed contents

function addPGain!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # correct gain for power
    gi = sqrt(g)

    # amplify/suppress signal
    x .*= gi

end



#   addAGain!
#
#   info:
#       This function adds gain to the amplitude of the signal by multiplying the amplitude with it.
#
#   input arguments:
#       x::AbstractArray        - Array of arbitrary shape with arbitrary contents
#       g::Real                 - Gain factor
#   output arguments:
#       _::AbstractArray        - Array of similar shape with amplified/suppressed contents

function addAGain!(x::AbstractArray{T,N}, g::Real) where {T,N}

    # amplify/suppress signal
    x .*= g

end



#   normalizeVar!
#
#   info:
#       This function normalizes an array by dividing itself by its variance.
#
#   input arguments:
#       a::AbstractArray        - Array of arbitrary shape with arbitrary contents
#   output arguments:
#       _::AbstractArray        - Array of similar shape with mean-corrected contents

function normalizeVar!(a::AbstractArray{T,N}) where {T,N}

    # calculate variance
    varvalue = var(a)

    # normalize by variance
     a ./= varvalue

end



#   normalizeStd!
#
#   info:
#       This function normalizes an array by dividing itself by its standard deviation.
#
#   input arguments:
#       a::AbstractArray        - Array of arbitrary shape with arbitrary contents
#   output arguments:
#       _::AbstractArray        - Array of similar shape with mean-corrected contents

function normalizeStd!(a::AbstractArray{T,N}) where {T,N}

    # calculate standard deviation
    stdvalue = std(a)

    # normalize by standard deviation
     a ./= stdvalue

end



#   subtractMean!
#
#   info:
#       This function subtracts the mean value from an array.
#
#   input arguments:
#       a::AbstractArray        - Array of arbitrary shape with arbitrary contents
#   output arguments:
#       _::AbstractArray        - Array of similar shape with mean-corrected contents

function subtractMean!(a::AbstractArray{T,N}) where {T,N}

    # calculate mean value
    meanvalue = mean(a)

    # subtract mean value
     a .-= meanvalue

end


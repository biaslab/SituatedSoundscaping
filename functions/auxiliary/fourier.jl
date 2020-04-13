# Information:
# This script contains several functions related to the Fourier transformation.
#

function calc_C(f::Array{Float64}, t::Array{Float64})
# Info:     This function calculates the Fourier series matrix C, which
#           decomposes a time-domain signal into its real Fourier coefficients.
#
# Inputs:   f - array with center frequencies of Fourier coefficients
#           t - time stamps for sinusoidal functions in the C-matrix
#
# Outputs:  C - Matrix of dimensions (N, 2M+1), where N represents the amount of
#               time instances and where M represents the amount of frequency
#               bins, which creates the Fourier decomposition

    # allocate space for matrix C
    C = Array{Float64}(undef, length(t), 2*length(f)+1)

    # loop through time (rows) and fill matrix C
    for (idx, ti) in enumerate(t)
        C[idx, :] = cat(dims=1, [1.0], sin.(2*pi*f*ti), cos.(2*pi*f*ti))
    end

    # return matrix C
    return C
end;
calc_C(f::Array{Int64}, t::Array{Float64}) = calc_C(convert(Float64, f), t)

function cartesian2polar(a, b)
# Info:     This function transforms complex numbers in cartesian notation to
#           polar notation.
#
# Inputs:   a - real parts of complex number(s)
#           b - imaginary parts of complex number(s)
#
# Outputs:  magnitude - magnitude of complex number
#           phase -     phase of complex number

    # calculate magnitude
    magnitude = sqrt.(a.^2 + b.^2)

    # calculate phase
    phase = angle.(a - 1im*b)

    # return outputs
    return magnitude, phase
end

function polar2cartesian(magnitude, phase)
# Info:     This function transforms complex numbers in polar notation to
#           cartesian notation.
#
# Inputs:   magnitude - magnitude of complex number
#           phase -     phase of complex number
#
# Outputs:  a - real parts of complex number(s)
#           b - imaginary parts of complex number(s)

    # calculate real part
    a = magnitude.*cos.(phase)

    # calculate imaginary part
    b = -magnitude.*sin.(phase)

    # return outputs
    return a, b
end

function FFTovertime(signal::Array{Float64}, len::Int64, overlap::Int64, window; pad=0)
# Info: This function calculates the FFT of a signal of blocks of a certain
#       length with a specified overlap and windowing function and possibly
#       zero-padding for an increased spectral resolution.
#
# Inputs:   signal -    time-domain signal to transform
#           len -       length of the windowed signal
#           overlap -   amount of samples of overlap between consecutive windows
#           window -    windowing function
#           [pad] -     amount of zero-padding
#
# Outputs:  S - Matrix containing windowed transforms

    # abbreviate inputs for simplicity
    s = signal
    l = len
    o = overlap
    w = window
    # not slow, but fast fourier transform

    # calculate the amount of possible windows
    nr_windows = (length(signal)-l)÷(l-o)

    # create placeholder for fft results
    S = Array{Complex{Float64}, 2}(undef, nr_windows, l+pad)

    # fill array
    for k = 1:nr_windows

        # calculate FFT
        if pad == 0
            # window signal without zero-padding
            x = s[1+(k-1)*(l-o):l+(k-1)*(l-o)].*window(l)
        else
            # window signal with zero-padding
            x = vcat(s[1+(k-1)*(l-o):l+(k-1)*(l-o)].*window(l), zeros(pad))
        end

        # calculate Fourier coefficients and store
        S[k,:] = FFTW.fft(x)

    end

    # return outputs
    return S
end

function PSDovertime(signal::Array{Float64}, len::Int64, overlap::Int64, window; pad=0)
# Info: This function calculates the PSD of a signal of blocks of a certain
#       length with a specified overlap and windowing function and possibly
#       zero-padding for an increased spectral resolution.
#
# Inputs:   signal -    time-domain signal to transform
#           len -       length of the windowed signal
#           overlap -   amount of samples of overlap between consecutive windows
#           window -    windowing function
#           [pad] -     amount of zero-padding
#
# Outputs:  S - Matrix containing windowed transforms

    s = signal
    l = len
    o = overlap
    w = window

    # create placeholder for fft
    windows = (length(signal)-l)÷(l-o)
    S = Array{Float64, 2}(undef, windows, l+pad)

    # fill array
    for k = 1:windows

        # calculate PSD
        if pad == 0
            x = s[1+(k-1)*o:l+(k-1)*o].*window(l)
        else
            x = vcat(s[1+(k-1)*o:l+(k-1)*o].*window(l), zeros(pad))
        end
        S[k,:] = abs2.(FFTW.fft(x))

    end

    return S
end


function logPSDovertime(signal::Array{Float64}, len::Int64, overlap::Int64, window; pad=0)
# Info: This function calculates the logPSD of a signal of blocks of a
#       certain length with a specified overlap and windowing function and
#       possibly zero-padding for an increased spectral resolution.
#
# Inputs:   signal -    time-domain signal to transform
#           len -       length of the windowed signal
#           overlap -   amount of samples of overlap between consecutive windows
#           window -    windowing function
#           [pad] -     amount of zero-padding
#
# Outputs:  S - Matrix containing windowed transforms

    s = signal
    l = len
    o = overlap
    w = window
    # not slow, but fast fourier transform

    # create placeholder for fft
    windows = (length(signal)-l)÷(l-o)
    S = Array{Float64, 2}(undef, windows, l+pad)

    # fill array
    for k = 1:windows

        # calculate log PSD
        if pad == 0
            x = s[1+(k-1)*o:l+(k-1)*o].*window(l)
        else
            x = vcat(s[1+(k-1)*o:l+(k-1)*o].*window(l), zeros(pad))
        end
        S[k,:] = log.(abs2.(FFTW.fft(x)))

    end

    return S
end

function Cepstrumovertime(signal, len, overlap, window; pad=0)
# Info: This function calculates the cepstrum of a signal of blocks of a
#       certain length with a specified overlap and windowing function and
#       possibly zero-padding for an increased spectral resolution.
#
# Inputs:   signal -    time-domain signal to transform
#           len -       length of the windowed signal
#           overlap -   amount of samples of overlap between consecutive windows
#           window -    windowing function
#           [pad] -     amount of zero-padding
#
# Outputs:  S - Matrix containing windowed transforms

    s = signal
    l = len
    o = overlap
    w = window
    # not slow, but fast fourier transform

    # create placeholder for fft
    windows = (length(signal)-l)÷(l-o)
    S = Array{Float64, 2}(undef, windows, l+pad)

    # fill array
    for k = 1:windows

        # calculate log PSD
        if pad == 0
            x = s[1+(k-1)*o:l+(k-1)*o].*window(l)
        else
            x = vcat(s[1+(k-1)*o:l+(k-1)*o].*window(l), zeros(pad))
        end
        S[k,:] = FFTW.dct(log.(abs2.(FFTW.fft(x))))

    end

    return S
end

function singlesided(A)
# Info: This function calculates the lgoPSD of a signal of blocks of a
#       certain length with a specified overlap and windowing function and
#       possibly zero-padding for an increased spectral resolution.
#
# Inputs:   A - matrix containing two-sided results of transform
#
# Outputs:  A - matrix containing one-sided results of transform

    return A[:, 1:(size(A)[2]+1)÷2]

end

function rectangularwindow(length::Int64)
    return ones(length)./length
end

function triangularwindow(length::Int64)
    # computes a triangular (Bartlett) window
    if mod(length, 2) == 1
        return vcat(collect(1:div(length, 2))/div(length+2, 2), 1, collect(div(length, 2):-1:1)/div(length+2, 2))
    else
        return vcat(collect(1:div(length, 2))/div(length+2, 2), collect(div(length, 2):-1:1)/div(length+2, 2))
    end
end

function sinewindow(length::Int64)
    # computes a sine window
    return sin.(pi/length*collect(1:length))
end

function welchwindow(length::Int64)
    return 1 .- ((collect(1:length) .- length/2)/(length/2)).^2
end

function hanningwindow(length::Int64)
    a0 = 0.5
    return a0 .- (1-a0)*cos.(2*pi*collect(1:length)/length)
end

function hammingwindow(length::Int64)
    a0 = 25/46
    return a0 .- (1-a0)*cos.(2*pi*collect(1:length)/length)
end

function blackmanwindow(length::Int64)
    alpha = 0.16
    a0 = (1-alpha)/2
    a1 = 0.5
    a2 = alpha/2

    return a0 .- a1*cos.(2*pi*collect(1:length)/length) + a2*cos.(4*pi*collect(1:length)/length)
end

function blackmannutallwindow(length::Int64)
    a0 = 0.3635819
    a1 = 0.4891775
    a2 = 0.1365995
    a3 = 0.0106411

    return a0 .- a1*cos.(2*pi*collect(1:length)/length) + a2*cos.(4*pi*collect(1:length)/length) - a2*cos.(6*pi*collect(1:length)/length)
end

function plot_spectrogram(spec, fs; ax="none", fontsize=10, sparse=false)
# Info: This function creates a spectrogram
#
# Inputs:  spec -       calculated spectrum
#          fs   -       sampling frequency
#          [ax] -       axes to plot on
#          [fontsize] - fontsize of axes
#          [sparse] -   whether to sparsify the plot
#
# Outputs: None

    # create cartoon font
    # xkcd()

    # plot spectrogram
    if ax == "none"
        x = spec.power .- minimum(spec.power)
        if sparse
            y = 1e-11
            for k = 1:100
                x = soft_thresholding2d(x, y);
            end
        end
        imshow(reverse(log10.(x), dims=1),
               aspect="auto",
               cmap="jet",
               origin="lower",
               extent=[first(spec.time), last(spec.time), last(spec.freq), first(spec.freq)])

        # set limits of axes
        xlim(first(spec.time), last(spec.time))
        ylim(first(spec.freq), last(spec.freq))

        # set axes labels
        xlabel("time [sec]", fontsize=fontsize)
        ylabel("frequency [Hz]", fontsize=fontsize)

        # set ticks
        tick_params(labelsize=fontsize)
        
        # create colorbar
        colorbar()

    else

        # create plot
        x = spec.power .- minimum(spec.power)
        if sparse
            y = 1e-11
            for k = 1:100
                x = soft_thresholding2d(x, y);
            end
        end
        ax.imshow(reverse(log10.(x), dims=1),
               aspect="auto",
               cmap="jet",
               origin="lower",
               extent=[first(spec.time), last(spec.time), last(spec.freq), first(spec.freq)])

        # set limits of axes
        ax.set_xlim(first(spec.time), last(spec.time))
        ax.set_ylim(first(spec.freq), last(spec.freq))

        # set axes labels
        ax.set_xlabel("time [sec]", fontsize=fontsize)
        ax.set_ylabel("frequency [Hz]", fontsize=fontsize)

        # set ticks
        ax.tick_params(labelsize=fontsize)

    end


end

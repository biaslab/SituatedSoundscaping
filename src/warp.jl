using LinearAlgebra: I, diagm

mutable struct warped_filter_bank
    block_duration_s::Float64
    calculate_gain::Function
    nr_bands::Int64
    postprocess_signal::Function
    preprocess_signal::Function
    sample_rate_hz::Int64

    block_length::Int64
    nr_fft::Int64
    all_pass_coefficient::Float64
    analysis_window::Array{Float64,1}               # n_fft x 1 vector
    synthesis_matrix::Array{Float64,2}    # (n_fft / 2) x n_bands
    fir_weights::Array{Float64,1}                   # (n_fft / 2) x 1
    # gain_db::Array{Float64, 2}                      # 1 x n_bands
    taps::Array{Float64,2}                         # n_fft x (n_fft / 2)

    # warped filter bank inner constructor
    function warped_filter_bank(;   block_duration_s::Float64=0.002,
                                    calculate_gain::Function=(gain_db, power_db) -> gain_db,
                                    nr_bands::Int64=17,
                                    postprocess_signal::Function=(samples) -> samples,
                                    preprocess_signal::Function=(samples) -> samples,
                                    sample_rate_hz::Int64=16000)

        # Calculate intermediate parameters
        block_length = calculate_block_length(block_duration_s, sample_rate_hz)
        nr_fft = calculate_nr_fft(nr_bands)
        all_pass_coefficient = calculate_all_pass_coefficient(sample_rate_hz)
        analysis_window = calculate_analysis_window(block_length)
        synthesis_matrix = calculate_synthesis_matrix(nr_bands, nr_fft)
        fir_weights = calculate_fir_weights(synthesis_matrix)
        taps = calculate_taps(block_length, nr_fft)

        # Initialize filterbank
        filterbank = new(block_duration_s, 
                         calculate_gain, 
                         nr_bands,
                         postprocess_signal, 
                         preprocess_signal, 
                         sample_rate_hz,
                         block_length,
                         nr_fft,
                         all_pass_coefficient,
                         analysis_window,
                         synthesis_matrix,
                         fir_weights,
                         taps)

        # return filter bank
        return filterbank

    end

end

function calculate_block_length(block_duration_s::Float64, sample_rate_hz::Int64)
    # This function calculates the length of an audio block.
    return Int(block_duration_s * sample_rate_hz)
end

function calculate_nr_fft(nr_bands::Int64)
    # This function calculates the length of the FFT window based on the desired number of frequency bands.
    return (nr_bands - 1) * 2
end

function calculate_all_pass_coefficient(sample_rate_hz::Int64)
    # This function calculates the all pass coefficients that most closely approximate the Bark scale for a given sampling frequency.
    return 0.8517 * sqrt(atan(0.06583 * (0.001 * sample_rate_hz))) - 0.1916
end

function calculate_analysis_window(nr_fft::Int64)
    # This function returns the Hanning window and its mirrored variant.
    window_half = 0.5 * (1 .- cos.(2 * pi * (1:nr_fft / 2) / (nr_fft + 1)))
  
    # Normalize the maximum coefficient to 1 and extend symmetrically
    window_half = window_half / maximum(window_half)
    window_full = [window_half; reverse(window_half, dims=1)]

    return window_full
end

function calculate_synthesis_matrix(nr_bands::Int64, nr_fft::Int64)
    nr_half = Int64(nr_fft / 2)
  
    # Create inverse FFT matrix
    synthesis_matrix = real(DSP.fft(eye(nr_fft), 1)) / nr_fft
  
    # Extend the synthesis matrix to even symmetry
    extender_matrix = [eye(nr_bands); reverse(eye(nr_bands), dims=1)[2:Int64(nr_bands) - 1, :]]
    synthesis_matrix = synthesis_matrix * extender_matrix
  
    # Center non-unique filter coefficients around the middle tap
    synthesis_matrix = reverse(eye(nr_half), dims=2) * synthesis_matrix[1:nr_half, :]
  
    # Currently only the Hann window is supported. This could be extended to
    # arbitrary windows of the correct size
    synthesis_window = 0.5 * (1 .- cos.(2 * pi * (1:nr_half) / nr_fft))
  
    # Normalize the maximum coefficient to 1 and extend symmetrically
    synthesis_window = synthesis_window / maximum(synthesis_window)
  
    # Apply the synthesis window to the basis vectors
    diagm(synthesis_window) * synthesis_matrix

end

function calculate_fir_weights(synthesis_matrix::Array{Float64,2})
    return synthesis_matrix * ones(size(synthesis_matrix,2))
end

function calculate_fir_weights(synthesis_matrix::Array{Float64,2}, G::Array{Float64,1})
    return synthesis_matrix * G
end

function update_fir_weights!(filterbank::warped_filter_bank, G::Array{Float64,1})

    # If n_bands gets updated, the situation can occur where there is a dimension
    # mismatch between gain_db and synthesis_matrix, as both depend on n_bands
    # but aren't updated simultaneously. If this case occurs the update is skipped
    if length(G) == size(filterbank.synthesis_matrix, 2)
        # The gain_db profile should be updated with a system calibration vector, but
        # this can be set to 0 for practical purposes on a computer and is therefore
        # omitted
        fir_weights = vec(filterbank.synthesis_matrix * G)
        filterbank.fir_weights = fir_weights
    end

end

function calculate_taps(block_length::Int64, nr_fft::Int64) 
    return zeros(Float64, nr_fft, block_length)
end

function filterTaps(all_pass_coefficient::Float64, taps::Array{Float64,2}, x::Array{Float64,1})
    n_fft_bins = size(taps, 1)
    n_taps = size(taps, 2)
  
    # When processing the first tap take the last tap from the previous time step
    # (i.e. when the function was last called on this set of taps) into account
    previous_tap_index = n_taps
  
    for tap_index = 1:n_taps
        taps[1, tap_index] = x[tap_index]
  
        for bin_index = 2:n_fft_bins
            tap_diff = taps[bin_index, previous_tap_index] - taps[bin_index - 1, tap_index]
  
            taps[bin_index, tap_index] = taps[bin_index - 1, previous_tap_index] +
            all_pass_coefficient * tap_diff
        end
  
        previous_tap_index = tap_index
    end
  
    return taps
end

function get_power(filterbank::warped_filter_bank)

    windowed_taps = filterbank.analysis_window .* filterbank.taps[:, end]
  
    fft_taps = DSP.fft(windowed_taps)[1:filterbank.nr_bands]'
  
    #power_db = 10*log10.(abs2.(fft_taps))
    power_logpower = log.(abs2.(fft_taps))

    #power_db = map(max, power_db, -30*ones(filterbank.nr_bands))
    
    return power_logpower
end

function get_frequency_coefficients(filterbank::warped_filter_bank; windowed::Bool=true)
    if windowed
        windowed_taps = filterbank.analysis_window .* filterbank.taps[:, end]
    else
        windowed_taps = filterbank.taps[:, end]
    end
    fft_taps = DSP.fft(windowed_taps)[1:filterbank.nr_bands]'
    return fft_taps
end

function calculate_output(fir_weights::Array{Float64,1}, taps::Array{Float64,2})
    # Symmetric extension of the unique warp filter coefficients
    fir_weights_full = [fir_weights; fir_weights[end - 1:-1:1]]
    return taps' * fir_weights_full
end
calculate_output(filterbank::warped_filter_bank) = calculate_output(filterbank.fir_weights, filterbank.taps[1:end - 1, :])

function Base.read( filterbank::warped_filter_bank;
                    postprocess_signal = filterbank.postprocess_signal)
    return postprocess_signal(calculate_output(filterbank))
end

function write!(filterbank::warped_filter_bank, x::Array{Float64,1};
                #calculate_gain=filterbank.calculate_gain,
                preprocess_signal=filterbank.preprocess_signal)
    # By updating the delay line before determining the power, gain can be
    # applied more instantaneously to the input signal. A side-effect is that a
    # small amount of non-causality is introduced as the gain is determined using
    # the entire block, but already applied to the first sample of the block.
    filterbank.taps = filterTaps(filterbank.all_pass_coefficient, filterbank.taps,
                        preprocess_signal(x))

    #power_db = getPower(filterbank)
    #setParam!(filterbank, :gain_db, calculate_gain(filterbank.gain_db, power_db))
end

function run!(filterbank::warped_filter_bank, x::Array{Float64,1};
             postprocess_signal = filterbank.postprocess_signal,
             preprocess_signal = filterbank.preprocess_signal)
        write!(filterbank, x,
               preprocess_signal = preprocess_signal)
        read(filterbank, postprocess_signal = postprocess_signal)
end

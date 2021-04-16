export separate_sources_wiener 

function separate_sources_wiener(x, s, n; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5)

    # calculate number of blocks to process
    nr_blocks = Int(length(x)/block_length)
    nr_freqs = block_length ÷ 2 + 1

    # initialize warped filter bank
    filterbank_x = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)
    filterbank_s = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)
    filterbank_n = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)

    # allocate some space
    X = Array{Complex{Float64},2}(undef, nr_blocks, nr_freqs)
    S = Array{Complex{Float64},2}(undef, nr_blocks, nr_freqs)
    N = Array{Complex{Float64},2}(undef, nr_blocks, nr_freqs)
    G = Array{Float64,2}(undef, nr_blocks, nr_freqs)
    output = zeros(size(x))

    # loop through blocks
    @showprogress for k in 1:nr_blocks

        # feed signals into filterbanks
        run!(filterbank_x, x[1+(k-1)*block_length:k*block_length])
        run!(filterbank_s, s[1+(k-1)*block_length:k*block_length])
        run!(filterbank_n, n[1+(k-1)*block_length:k*block_length])

        # extract frequency coefficients
        X[k,:] = squeeze(get_frequency_coefficients(filterbank_x))
        S[k,:] = squeeze(get_frequency_coefficients(filterbank_s))
        N[k,:] = squeeze(get_frequency_coefficients(filterbank_n))

        # calculate  gain
        G[k,:] = abs2.(S[k,:]) ./ (abs2.(S[k,:]) + abs2.(N[k,:]))

        # update filter fir_weights
        update_fir_weights!(filterbank_x, G[k,:])

        # read from filter
        output[1+(k-1)*block_length:k*block_length] = read(filterbank_x)

    end

    return output, X, S, N, G

end
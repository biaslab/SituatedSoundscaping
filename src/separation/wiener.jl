export separate_sources_wiener 

function separate_sources_wiener(folder, x, s, n; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5, power_dB::Real=0, save_results::Bool=false)

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

    if save_results
        folder_extended = string("_power=", power_dB)
        plot_wiener(folder, folder_extended, G, output, block_length, fs)
    end

    return output, X, S, N, G

end


function plot_wiener(folder, folder_extended, G, output, block_length, fs)
    
    # plot text
    filterbank = warped_filter_bank(block_duration_s=block_length/fs, nr_bands=Int(block_length/2)+1)
    nr_blocks = Int(length(output)/block_length)
    nr_freqs = block_length ÷ 2 + 1
    X = zeros(nr_freqs,nr_blocks)
    for k in 1:nr_blocks

        # feed signals into filterbanks
        run!(filterbank, output[1+(k-1)*block_length:k*block_length])

        # read from filter
        X[:,k] = log.(abs2.(squeeze(get_frequency_coefficients(filterbank))))

    end

    # plot gain
    plt.figure()
    plt.imshow(X, aspect="auto", origin="lower", cmap="jet")
    plt.xlabel("frame")
    plt.ylabel("frequency bin")
    plt.colorbar()
    plt.gcf()
    plt.savefig(folder*"/warped_spectrum"*folder_extended*".eps") 
    plt.savefig(folder*"/warped_spectrum"*folder_extended*".png") 

    # plot gain
    plt.figure()
    plt.imshow(G', aspect="auto", origin="lower", cmap="jet")
    plt.xlabel("frame")
    plt.ylabel("frequency bin")
    plt.colorbar()
    plt.gcf()
    plt.savefig(folder*"/gain"*folder_extended*".eps") 
    plt.savefig(folder*"/gain"*folder_extended*".png") 

    # plot logpower gain
    plt.figure()
    plt.imshow(log.(abs2.(G))', aspect="auto", origin="lower", cmap="jet")
    plt.xlabel("frame")
    plt.ylabel("frequency bin")
    plt.colorbar()
    plt.gcf()
    plt.savefig(folder*"/logpowergain"*folder_extended*".eps") 
    plt.savefig(folder*"/logpowergain"*folder_extended*".png") 

    # plot signal
    plt.figure()
    plt.plot(collect(1:length(output))/fs, output)
    plt.grid()
    plt.xlabel("time [sec]")
    plt.gcf()
    plt.savefig(folder*"/output_signal"*folder_extended*".eps") 
    plt.savefig(folder*"/output_signal"*folder_extended*".png") 

    # save signal
    wavwrite(normalize_range(output), folder*"/output_signal"*folder_extended*".wav", Fs=fs)


end
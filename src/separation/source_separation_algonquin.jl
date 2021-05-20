using PyPlot, WAV, HDF5

export separate_sources_algonquin

function separate_sources_algonquin(folder, x, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5, power_dB::Real=0, save_results::Bool=true, nr_iterations::Int64=10)

    # calculate number of blocks to process
    nr_blocks = Int(length(x)/block_length)

    # initialize warped filter bank
    filterbank = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)

    # allocate some space
    X = Array{Float64,1}(undef, Int(block_length/2)+1)
    G = Array{Float64,2}(undef, nr_blocks, Int(block_length/2)+1)
    output = zeros(size(x))

    # loop through blocks
    @showprogress for n in 1:nr_blocks

        # feed signal into filterbank
        run!(filterbank, x[1+(n-1)*block_length:n*block_length])

        # extract log-power  coefficients
        X .= log.(abs2.(squeeze(get_frequency_coefficients(filterbank))))

        # calculate weighted gain
        G[n,:] = loop_inference_algonquin(X, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise, observation_noise_precision, nr_iterations)

        # update filter fir_weights
        update_fir_weights!(filterbank, G[n,:])

        # read from filter
        output[1+(n-1)*block_length:n*block_length] = read(filterbank)

    end

    # save files
    if save_results
        nr_frequencies = size(means_speech,1)
        nr_mixtures_speech = size(means_speech,2)
        nr_mixtures_noise = size(means_noise,2)
        folder_extended = "_freq="*string(nr_frequencies)*"_mixs="*string(nr_mixtures_speech)*"_mixn="*string(nr_mixtures_noise)*"_power="*string(power_dB)
        plot_algonquin(folder, folder_extended, G, output, block_length, fs)
    end

    # return output signal
    return output

end

# perform inference for all possible combinations
function loop_inference_algonquin(data, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise, observation_noise_precision, nr_iterations)

    # fetch items 
    nr_mixtures_speech = size(means_speech,2)
    nr_mixtures_noise = size(means_noise,2)

    # allocate space for free energy and gain vectors
    FE = Array{Float64,2}(undef, nr_mixtures_speech, nr_mixtures_noise)
    G = Array{Array{Float64,1},2}(undef, nr_mixtures_speech, nr_mixtures_noise)

    # loop over combinations
    for (ids, idn) in Iterators.product(1:nr_mixtures_speech, 1:nr_mixtures_noise)

        # do inference
        FE[ids, idn], G[ids, idn] = inference_algonquin(data, means_speech[:,ids], covs_speech[:,ids], means_noise[:,idn], covs_noise[:,idn], observation_noise_precision, nr_iterations)

    end

    # set NaNs in gains to 1 (i.e. no filtering)
    replace!.(G, NaN=>1.0)
    replace!(FE, NaN=>1e10)

    # determine posterior mixture probabilities (and remove nans)
    prior = w_speech * w_noise'
    @assert sum(prior) ≈ 1
    posterior = -FE .+ log.(prior)
    softmax_nan!(posterior)

    # weight gain by posteriors
    Gw = sum(G .* posterior)

    # return weighted gain
    return Gw

end

function inference_algonquin(data, mean_speech, cov_speech, mean_noise, cov_noise, observation_noise_precision, nr_iterations)

    # find number of frequencies
    nr_freqs = length(mean_speech)

    # create model
    model, (s, n, y) = source_separation_model_algonquin(nr_freqs, mean_speech, cov_speech, mean_noise, cov_noise, observation_noise_precision)

    # allocate space for marginals
    marg_s = keep(Vector{Marginal})
    marg_n = keep(Vector{Marginal})

    # allocate space for free energy
    fe = ScoreActor(Float64)

    # subscribe to marginals
    s_sub = subscribe!(getmarginals(s), marg_s)
    n_sub = subscribe!(getmarginals(n), marg_n)

    # subscribe to free energy
    fe_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    # set initial marginals
    for freq in 1:nr_freqs

        setmarginal!(s[freq], GaussianMeanPrecision(mean_speech[freq], 1/cov_speech[freq]))

        setmarginal!(n[freq], GaussianMeanPrecision(mean_noise[freq], 1/cov_noise[freq]))

    end

    # perform variational message passing
    for it in 1:nr_iterations

        # update data
        ReactiveMP.update!(y, data)

    end

    # unsubscribe from marginal streams
    unsubscribe!(s_sub)
    unsubscribe!(n_sub)

    # unsubscribe from score streams
    unsubscribe!(fe_sub)

    # fetch free energy and gain
    FE = getvalues(fe)[end]
    G = exp.(mean.(getvalues(marg_s)[end])) ./ (exp.(mean.(getvalues(marg_s)[end])) + exp.(mean.(getvalues(marg_n)[end])))

    # return free energy and gain vector
    return FE, G

end

# create model
@model [ default_factorisation = MeanField() ] function source_separation_model_algonquin(nr_freqs, ps_μ, ps_v, pn_μ, pn_v, observation_noise_precision)
    
    # allocate variables
    s = randomvar(nr_freqs)
    n = randomvar(nr_freqs)
    y = datavar(Float64, nr_freqs)

    # loop through frequencies
    for freq in 1:nr_freqs

        # create speech model
        s[freq] ~ GaussianMeanPrecision(ps_μ[freq], 1/ps_v[freq])

        # create noise model
        n[freq] ~ GaussianMeanPrecision(pn_μ[freq], 1/pn_v[freq])

        # specify observations
        y[freq] ~ Algonquin(s[freq], n[freq], observation_noise_precision)

    end

    # return random variables
    return s, n, y

end


# plot stuff
function plot_algonquin(folder, folder_extended, G, output, block_length, fs)
    
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

    # save gain
    f = h5open(folder*"/gain"*folder_extended*".h5", "w")
    HDF5.write(f, "gain", G);
    close(f)

end
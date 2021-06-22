using PyPlot, WAV, HDF5

export separate_sources_gs_sum_simplified

function separate_sources_gs_sum_simplified(folder, x, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5, power_dB::Real=0, save_results::Bool=true, nr_iterations::Int64=10)

    # calculate number of blocks to process
    nr_blocks = Int(length(x)/block_length)

    # initialize warped filter bank
    filterbank = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)

    # allocate some space
    X = Array{Complex{Float64},1}(undef, Int(block_length/2)+1)
    G = Array{Float64,2}(undef, nr_blocks, Int(block_length/2)+1)
    output = zeros(size(x))

    # loop through blocks
    @showprogress for n in 1:nr_blocks

        # feed signal into filterbank
        run!(filterbank, x[1+(n-1)*block_length:n*block_length])

        # extract frequency coefficients
        X = squeeze(get_frequency_coefficients(filterbank))

        # calculate weighted gain
        G[n,:] = loop_inference_gs_sum_simplified(X, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a, observation_noise_precision, nr_iterations)

        # update filter fir_weights
        update_fir_weights!(filterbank, G[n,:])

        # read from filter
        output[1+(n-1)*block_length:n*block_length] = read(filterbank)

    end

    # save files
    if save_results
        nr_frequencies = Int(block_length/2)+1
        nr_mixtures_speech = length(qs_a)
        nr_mixtures_noise = length(qn_a)
        folder_extended = "_freq="*string(nr_frequencies)*"_mixs="*string(nr_mixtures_speech)*"_mixn="*string(nr_mixtures_noise)*"_power="*string(power_dB)
        plot_gs_sum(folder, folder_extended, G, output, block_length, fs)
    end

    # return output signal
    return output

end

# perform inference for all possible combinations
function loop_inference_gs_sum_simplified(data, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a, observation_noise_precision, nr_iterations)

    # fetch items 
    nr_mixtures_speech = length(qs_μ)
    nr_mixtures_noise = length(qn_μ)

    # allocate space for free energy and gain vectors
    FE = Array{Float64,2}(undef, nr_mixtures_speech, nr_mixtures_noise)
    G = Array{Array{Float64,1},2}(undef, nr_mixtures_speech, nr_mixtures_noise)

    # loop over combinations
    for (ids, idn) in Iterators.product(1:nr_mixtures_speech, 1:nr_mixtures_noise)

        # do inference
        FE[ids, idn], G[ids, idn] = inference_gs_sum_simplified(data, qs_μ[ids], qs_γ[ids], qn_μ[idn], qn_γ[idn], observation_noise_precision, nr_iterations)

    end

    # determine posterior mixtures
    prior = softmax(logmean(qs_a)) * softmax(logmean(qn_a))'
    @assert sum(prior) ≈ 1
    posterior = -FE .+ log.(prior)
    softmax!(posterior)

    # weight gain by posteriors
    Gw = sum(G .* posterior)

    # return weighted gain
    return Gw

end


# do inference in model
function inference_gs_sum_simplified(data, qs_μ, qs_γ, qn_μ, qn_γ, observation_noise_precision, nr_iterations)

    # find number of frequencies
    nr_freqs = length(qs_μ)

    # create model
    model, (ξs, ξn, Y) = source_separation_model_gs_sum_simplified(nr_freqs, qs_μ, qs_γ, qn_μ, qn_γ, observation_noise_precision)

    # allocate space for marginals
    marg_ξs = keep(Vector{Marginal})
    marg_ξn = keep(Vector{Marginal})

    # allocate space for free energy
    fe = ScoreActor(Float64)

    # subscribe to marginals
    ξs_sub = subscribe!(getmarginals(ξs), marg_ξs)
    ξn_sub = subscribe!(getmarginals(ξn), marg_ξn)

    # subscribe to free energy
    fe_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    # set initial marginals
    for freq in 1:nr_freqs

        # setmarginal!(ξs_μ[freq], GaussianMeanPrecision(qs_μ.μ[freq], qs_μ.γ[freq]))
        # setmarginal!(ξs_γ[freq], GammaShapeRate(qs_γ.a[freq], qs_γ.b[freq]))
        setmarginal!(ξs[freq], GaussianMeanPrecision(qs_μ.μ[freq], qs_γ.a[freq]/qs_γ.b[freq]))

        # setmarginal!(ξn_μ[freq], GaussianMeanPrecision(qn_μ.μ[freq], qn_μ.γ[freq]))
        # setmarginal!(ξn_γ[freq], GammaShapeRate(qn_γ.a[freq], qn_γ.b[freq]))
        setmarginal!(ξn[freq], GaussianMeanPrecision(qn_μ.μ[freq], qn_γ.a[freq]/qn_γ.b[freq]))

    end

    # perform variational message passing
    for it in 1:nr_iterations

        # update data
        ReactiveMP.update!(Y, data)

    end

    # unsubscribe from marginal streams
    unsubscribe!(ξs_sub)
    unsubscribe!(ξn_sub)

    # unsubscribe from score streams
    unsubscribe!(fe_sub)

    # fetch free energy and gain
    FE = getvalues(fe)[end]
    G = exp.(mean.(getvalues(marg_ξs)[end])) ./ (exp.(mean.(getvalues(marg_ξs)[end])) + exp.(mean.(getvalues(marg_ξn)[end])))

    # return free energy and gain vector
    return FE, G

end

# create model
@model [ default_factorisation = MeanField() ] function source_separation_model_gs_sum_simplified(nr_freqs, ps_μ, ps_γ, pn_μ, pn_γ, observation_noise_precision)
    
    # allocate variables
    ξs = randomvar(nr_freqs)
    ξn = randomvar(nr_freqs)
    Y = datavar(Complex{Float64}, nr_freqs)

    # loop through frequencies
    for freq in 1:nr_freqs

        # create speech model
        ξs[freq] ~ GaussianMeanPrecision(ps_μ.μ[freq], ps_γ.a[freq]/ps_γ.b[freq])

        # create noise model
        ξn[freq] ~ GaussianMeanPrecision(pn_μ.μ[freq], pn_γ.a[freq]/pn_γ.b[freq])


        # specify observations
        Y[freq] ~ GaussianScaleSum(ξs[freq], ξn[freq])
    end

    # return random variables
    return ξs, ξn, Y

end



# plot stuff
function plot_gs_sum_simplified(folder, folder_extended, G, output, block_length, fs)
    
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
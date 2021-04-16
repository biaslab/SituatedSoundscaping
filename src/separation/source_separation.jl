export separate_sources

function separate_sources(x, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5)

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
        G[n,:] = loop_inference(X, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a, observation_noise_precision)

        # update filter fir_weights
        update_fir_weights!(filterbank, G[n,:])

        # read from filter
        output[1+(n-1)*block_length:n*block_length] = read(filterbank)

    end

    return output, G

end

# perform inference for all possible combinations
function loop_inference(data, qs_μ, qs_γ, qs_a, qn_μ, qn_γ, qn_a, observation_noise_precision)

    # fetch items 
    nr_mixtures_speech = length(qs_μ)
    nr_mixtures_noise = length(qn_μ)

    # allocate space for free energy and gain vectors
    FE = Array{Float64,2}(undef, nr_mixtures_speech, nr_mixtures_noise)
    G = Array{Array{Float64,1},2}(undef, nr_mixtures_speech, nr_mixtures_noise)

    # loop over combinations
    for (ids, idn) in Iterators.product(1:nr_mixtures_speech, 1:nr_mixtures_noise)

        # do inference
        FE[ids, idn], G[ids, idn] = inference(data, qs_μ[ids], qs_γ[ids], qn_μ[idn], qn_γ[idn], observation_noise_precision)

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
function inference(data, qs_μ, qs_γ, qn_μ, qn_γ, observation_noise_precision; nr_iterations=10)

    # find number of frequencies
    nr_freqs = length(qs_μ)

    # create model
    model, (ξs_μ, ξs_γ, ξs, Xs, ξn_μ, ξn_γ, ξn, Xn, Y) = source_separation_model(nr_freqs, qs_μ, qs_γ, qn_μ, qn_γ, observation_noise_precision)

    # allocate space for marginals
    marg_Xs = keep(Vector{Marginal})
    marg_Xn = keep(Vector{Marginal})

    # allocate space for free energy
    fe = ScoreActor(Float64)

    # subscribe to marginals
    Xs_sub = subscribe!(getmarginals(Xs), marg_Xs)
    Xn_sub = subscribe!(getmarginals(Xn), marg_Xn)

    # subscribe to free energy
    fe_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)

    # set initial marginals
    for freq in 1:nr_freqs

        setmarginal!(ξs_μ[freq], GaussianMeanPrecision(qs_μ.μ[freq], qs_μ.γ[freq]))
        setmarginal!(ξs_γ[freq], GammaShapeRate(qs_γ.a[freq], qs_γ.b[freq]))
        setmarginal!(ξs[freq], GaussianMeanPrecision(qs_μ.μ[freq], qs_γ.a[freq]/qs_γ.b[freq]))

        setmarginal!(ξn_μ[freq], GaussianMeanPrecision(qn_μ.μ[freq], qn_μ.γ[freq]))
        setmarginal!(ξn_γ[freq], GammaShapeRate(qn_γ.a[freq], qn_γ.b[freq]))
        setmarginal!(ξn[freq], GaussianMeanPrecision(qn_μ.μ[freq], qn_γ.a[freq]/qn_γ.b[freq]))

    end

    # perform variational message passing
    for it in 1:nr_iterations

        # update data
        ReactiveMP.update!(Y, data)

    end

    # unsubscribe from marginal streams
    unsubscribe!(Xs_sub)
    unsubscribe!(Xn_sub)

    # unsubscribe from score streams
    unsubscribe!(fe_sub)

    # fetch free energy and gain
    FE = getvalues(fe)[end]
    G = real.(mean.(getvalues(marg_Xs)[end]) ./ (mean.(getvalues(marg_Xs)[end]) + mean.(getvalues(marg_Xn)[end])))

    # return free energy and gain vector
    return FE, G

end

# create model
@model [ default_factorisation = MeanField() ] function source_separation_model(nr_freqs, ps_μ, ps_γ, pn_μ, pn_γ, observation_noise_precision)
    
    # allocate variables
    ξs_μ = randomvar(nr_freqs)
    ξs_γ = randomvar(nr_freqs)
    ξs = randomvar(nr_freqs)
    Xs = randomvar(nr_freqs)
    ξn_μ = randomvar(nr_freqs)
    ξn_γ = randomvar(nr_freqs)
    ξn = randomvar(nr_freqs)
    Xn = randomvar(nr_freqs)
    Y = datavar(Complex{Float64}, nr_freqs)

    # loop through frequencies
    for freq in 1:nr_freqs

        # create speech model
        ξs_μ[freq] ~ GaussianMeanPrecision(ps_μ.μ[freq], ps_μ.γ[freq])
        ξs_γ[freq] ~ Gamma(ps_γ.a[freq], ps_γ.b[freq])
        ξs[freq] ~ GaussianMeanPrecision(ξs_μ[freq], ξs_γ[freq])
        Xs[freq] ~ GS(ξs[freq])

        # create noise model
        ξn_μ[freq] ~ GaussianMeanPrecision(pn_μ.μ[freq], pn_μ.γ[freq])
        ξn_γ[freq] ~ Gamma(pn_γ.a[freq], pn_γ.b[freq])
        ξn[freq] ~ GaussianMeanPrecision(ξn_μ[freq], ξn_γ[freq])
        Xn[freq] ~ GS(ξn[freq])

        # specify observations
        Y[freq] ~ ComplexNormal(Xs[freq] + Xn[freq], 1/observation_noise_precision, 0.0)

    end

    # return random variables
    return ξs_μ, ξs_γ, ξs, Xs, ξn_μ, ξn_γ, ξn, Xn, Y

end
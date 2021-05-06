using PyPlot, WAV

export separate_sources_algonquin

function separate_sources_algonquin(folder, x, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise; block_length::Int64=64, fs::Int64=16000, observation_noise_precision::Float64=1e5, power_dB::Real=0)

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
        G[n,:] = loop_inference_algonquin(X, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise, observation_noise_precision, block_length)

        # update filter fir_weights
        update_fir_weights!(filterbank, G[n,:])

        # read from filter
        output[1+(n-1)*block_length:n*block_length] = read(filterbank)

    end

    # save files
    nr_frequencies = size(means_speech,1)
    nr_mixtures_speech = size(means_speech,2)
    nr_mixtures_noise = size(means_noise,2)
    folder_extended = "_freq="*string(nr_frequencies)*"_mixs="*string(nr_mixtures_speech)*"_mixn="*string(nr_mixtures_noise)*"_power="*string(power_dB)
    plot_algonquin(folder, folder_extended, G, output, block_length, fs)

    # return output signal
    return output

end

# perform inference for all possible combinations
function loop_inference_algonquin(data, means_speech, covs_speech, w_speech, means_noise, covs_noise, w_noise, observation_noise_precision, block_length)

    # fetch items 
    nr_mixtures_speech = size(means_speech,2)
    nr_mixtures_noise = size(means_noise,2)

    # allocate space for free energy and gain vectors
    score = Array{Float64,2}(undef, nr_mixtures_speech, nr_mixtures_noise)
    G = Array{Array{Float64,1},2}(undef, nr_mixtures_speech, nr_mixtures_noise)

    # loop over combinations
    for (ids, idn) in Iterators.product(1:nr_mixtures_speech, 1:nr_mixtures_noise)

        # do inference
        score[ids, idn], G[ids, idn] = inference_algonquin(data, means_speech[:,ids], covs_speech[:,ids], means_noise[:,idn], covs_noise[:,idn], observation_noise_precision, block_length)

    end

    # determine posterior mixtures
    prior = w_speech * w_noise'
    @assert sum(prior) ≈ 1
    posterior = score .+ log.(prior)
    softmax!(posterior)
    
    # weight gain by posteriors
    Gw = sum(G .* posterior)

    # return weighted gain
    return Gw

end

function inference_algonquin(data, mean_speech, cov_speech, mean_noise, cov_noise, observation_noise_precision, block_length)
    x = mean_speech
    n = mean_noise
    Φ = vcat(cov_speech, cov_noise)
    Σ = vcat(cov_speech, cov_noise)
    Ψ = ones(Int(block_length/2+1))/observation_noise_precision
    μ = vcat(mean_speech, mean_noise)
    y = data
    
    gi = zeros(Int(block_length/2+1))
    gi_derivative = zeros(Int(block_length+2))
    
    g!(gi, x, n, block_length)
    g_derivative!(gi_derivative, x, n, block_length)
    
    for it = 1:5
        update!(y, gi, gi_derivative, Φ, x, n, μ, Σ, Ψ, block_length)
    end
    # calculate gain
    G = exp.(x) ./ (exp.(x) + exp.(n))

    # score model
    score = score_algonquin(y, gi, gi_derivative, Φ, x, n, μ, Σ, Ψ, block_length)

    # return
    return score, G
end

function g!(gi::Array{Float64,1}, x::Array{Float64,1}, n::Array{Float64,1}, block_length::Int64)
    @inbounds for k = 1:Int(block_length/2+1)
        gi[k] = x[k] + log(1 + exp(n[k]-x[k]))
    end
end

function g_derivative!(gi_derivative::Array{Float64,1}, x::Array{Float64,1}, n::Array{Float64,1}, block_length::Int64)
    @inbounds for k = 1:Int(block_length/2+1)
        tmp = exp(n[k]-x[k])
        gi_derivative[k] = 1 / (1 + tmp)
        gi_derivative[k+Int(block_length/2+1)] = gi_derivative[k] * tmp
    end
end

function update!(y::Array{Float64,1}, gi::Array{Float64,1}, gi_derivative::Array{Float64,1}, Φ::Array{Float64,1}, x::Array{Float64,1}, n::Array{Float64,1}, μ::Array{Float64,1}, Σ::Array{Float64,1}, Ψ::Array{Float64,1}, block_length::Int64)
    # update estimate for y
    g!(gi, x, n, block_length)

    # update derivative in point
    g_derivative!(gi_derivative, x, n, block_length)

    # update points
    for k = 1:Int(block_length/2+1)

        # speech elements
        Φ[k] = 1 / (1 / Σ[k] + gi_derivative[k]^2 / Ψ[k] )
        x[k] += Φ[k]*((μ[k] - x[k])/Σ[k] + gi_derivative[k]*(y[k] - gi[k])/Ψ[k])

        # noise elements
        Φ[k+Int(block_length/2+1)] = 1 / (1 / Σ[k+Int(block_length/2+1)] + gi_derivative[k+Int(block_length/2+1)]^2 / Ψ[k] )
        n[k] += Φ[k+Int(block_length/2+1)]*((μ[k+Int(block_length/2+1)] - n[k])/Σ[k+Int(block_length/2+1)] + gi_derivative[k+Int(block_length/2+1)]*(y[k] - gi[k]) / Ψ[k])

    end

end

function score_algonquin(y::Array{Float64,1}, gi::Array{Float64,1}, gi_derivative::Array{Float64,1}, Φ::Array{Float64,1}, x::Array{Float64,1}, n::Array{Float64,1}, μ::Array{Float64,1}, Σ::Array{Float64,1}, Ψ::Array{Float64,1}, block_length::Int64)
    
    # update estimate for y
    g!(gi, x, n, block_length)

    # update derivative in point
    g_derivative!(gi_derivative, x, n, block_length)

    # calculate temporary term
    tmp1 = 0
    tmp2 = 0
    for k = 1:Int(block_length/2+1)
        tmp1 += (x[k]-μ[k])^2/Σ[k]
        tmp1 += (n[k]-μ[k+Int(block_length/2+1)])^2/Σ[k+Int(block_length/2+1)]

        tmp2 += gi_derivative[k]^2/Ψ[k]*Φ[k]
        tmp2 += gi_derivative[k+Int(block_length/2+1)]^2/Ψ[k]*Φ[k+Int(block_length/2+1)]
    end

    return -0.5*sum(log.(Σ)) +
            0.5*sum(log.(Φ)) - 
            0.5*sum((y - gi).^2 ./ Ψ) -
            0.5*sum(Φ ./ Σ) -
            0.5*tmp1 - 
            0.5*tmp2

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


end
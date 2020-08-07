
function separate_sources(X::Array{Complex{Float64},2}, models::Array{GSMM{Float64},1}, σ2_noise::Float64)
    # only work when dealing with two models
    length(models) == 2 || error("The algorithm currently only works for two sources. The algorithm should be generalized (using @nloops for example) to work for an arbitrary number of sources. Also the message schedule should then be updated.")

    # determine number of sources
    nr_sources = length(models)

    # allocate array for separated signals
    X_sep = [Array{Complex{Float64},2}(undef, size(X)) for _=1:length(models)]

    # allocate array for posterior class probabilities
    log_p = Array{Float64,length(models)}(undef, [models[k].n for k=1:length(models)]...)

    # allocate arrays for upward messages from addition node
    message_upwards_addition_mean = [Array{Complex{Float64},1}(undef, size(X,2)) for _=1:length(models)]
    message_upwards_addition_var = [Array{Complex{Float64},1}(undef, size(X,2)) for _=1:length(models)]

    # loop through segments
    for n = 1:size(X,1)

        # calculate posterior class probabilities
        update_class_log_probabilities!(X[n,:], models, log_p) 
        p = convert_class_log_probabilities!(log_p) 

        # calculate posterior states of distinct signals
        messages_downwards_GSMM_var = calculate_messages_GSMM(models, p)

        # message from observation upwards
        message_upwards_observation_mean = X[n,:]
        message_upwards_observation_var = σ2_noise*ones(size(X,2))

        # calculate upward messages from addition 
        for k = 1:length(models)
            message_upwards_addition_mean[k] = message_upwards_observation_mean #- sum(messages_downwards_GSMM_mean[1:end .!= k]) # uncomment for multiple iterations
            message_upwards_addition_var[k] = message_upwards_observation_var + sum( messages_downwards_GSMM_var[1:end .!= k] )
        end

        # calculate marginals (means only)
        for k = 1:length(models)
            X_sep[k][n,:] = (message_upwards_addition_mean[k].*messages_downwards_GSMM_var[k])./(message_upwards_addition_var[k] .+ messages_downwards_GSMM_var[k])
        end

    end 

    # expanded single-sided spectrum 
    X_sepi = singlesided2twosided.(X_sep)
    

    # reconstruct signal
    x_sep = reconstruct_signal(X_sepi)

    return x_sep, X_sep

end


function reconstruct_signal(X_sep::Array{Array{Complex{Float64},2},1})

    # allocate array for speech signal
    x_sep = Array{Array{Float64,1},1}(undef, length(X_sep))

    # loop through separated signals
    for k = 1:length(X_sep)
        # get sizes
        block_length = size(X_sep[k],2)
        block_step = block_length ÷ 2

        # initialize arrays with zeros
        x_sep[k] = zeros(Int((prod(size(X_sep[k])) + block_length)/2))
        
        # overlap-add
        for ki = 1:size(X_sep[k],1)
            x_sep[k][(ki-1)*block_step+1:(ki-1)*block_step+block_length] = x_sep[k][(ki-1)*block_step+1:(ki-1)*block_step+block_length] + real.(DSP.ifft(X_sep[k][ki,:]))
        end
    end

    return x_sep

end

function singlesided2twosided(x::Array{Complex{Float64}, 2})
    if size(x,2)%2 == 1
        return hcat(zeros(size(x,1), 1), x[:,1:end], zeros(size(x,1), 1), conj.(reverse(x, dims=2)))
    else
        
    end
end

function calculate_messages_GSMM(models::Array{GSMM{Float64},1}, p::Array{Float64,2})

    # allocate arrays for variance of messages
    GSMM_v = Array{Array{Float64,1},1}(undef, length(models))

    for k = 1:length(models)
        # marginalized posterior probabilities
        pk = squeezesum(p,k)::Array{Float64,1}
        
        # downward messages
        GSMM_v[k] = 1 ./ ((1 ./ models[k].Σf) * pk) ::Array{Float64,1}

    end

    # return message variances
    return GSMM_v

end


function squeezesum(p::Array{Float64,N}, k::Int64)::Array{Float64,1} where {N}
    size_p = [size(p)...]
    size_p[1:end .!= k] .= 1
    tmp = ones(size_p...)
    sump = sum!(tmp,p)::Array{Float64,N}
    return squeeze(sump)::Array{Float64,1}
end


function convert_class_log_probabilities!(log_p::Array{Float64,N}) where {N}

    log_p .-= maximum(log_p)
    e_log_p = exp.(log_p)
    p = e_log_p ./ sum(e_log_p)

    return p
end


function update_class_log_probabilities!(X::Array{Complex{Float64},1}, models::Array{GSMM{Float64},1}, log_p::Array{Float64,N}) where {N}
    
    # loop through items
    Threads.@threads for k2 = 1:size(log_p,2)
        for k1 = 1:size(log_p,1)
            @inbounds log_p[k1,k2] = log(models[1].w[k1]) + log(models[2].w[k2]) + sum(ComplexNormal_logpdf_zm.(X, models[1].Σf[:,k1] + models[2].Σf[:,k2]))
        end
    end

end



function Normal_logpdf(x::Real, μ::Real, σ2::Real)
    # get standard deviation
    σ = sqrt(σ2)
    
    # calculate z
    z = (x - μ)/σ2

    # calculate log probability
    log_p = -(abs2(z) + log(2π))/2 - log(σ)

    # return log probability
    return log_p
end

function ComplexNormal_logpdf_zm(x::Complex{Float64}, σ2::Real)

    # calculate log probability
    log_p = -log(π)-log(σ2)-abs2(x)/σ2

    # return log probability
    return log_p

end

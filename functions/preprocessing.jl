function normalize(x::Array{Float64, 1}, norm::String)
    # normalizes signal
    if norm == "max"
        return (x .- mean(x)) ./ maximum(x)
    elseif norm == "std"
        return (x .- mean(x)) ./ std(x)
    elseif norm == "var"
        return (x .- mean(x)) ./ var(x)
    end
end

function μ_law_companding(x::Array{Float64, 1}, μ::Int64)
    # perform μ-law companding
    return sign.(x) .* log.(1 .+ μ*abs.(x)) / log(1 + μ)
end

function μ_law_expansion(x::Array{Float64, 1}, μ::Int64)
    # perform μ-law expansion
    return sign.(x) .* (1 / μ) .* ( (1 .+ μ).^abs.(x) .- 1 )
end

function allpass_update_matrix2(order::Int, z::Float64)
    # this function calculates the large update matrix for an all-pass filter
        
    # specify matrices
    A = [z 0
         1 0]
    B = [-z 1
         0  0]
    C = [0 0
         1 0]
    D = zeros(2,1)
    D[1] = 1
    
    # create matrix T
    T = [((k<=l) ? B^(l-k) : 0)*((k==1) ? C : A) for l=1:order, k=1:order]
    T = hvcat(order, permutedims(T,[2,1])...)
    
    # create matrix u
    #u = repeat(D, order)
    u = [B^(k-1)*D for k=1:order]
    u = vcat(u...)
    
    return T, u
end

function warp_fft(signal::Array{Float64, 1}, z::Float64, len::Int64; step_size::Int64=1)
    # get update matrices
    T, u = allpass_update_matrix2(len, z)
    
    # create current hidden states of filter
    Y = zeros(2*len)
    
    # create array for outputs
    y = Array{Complex{Float64}, 2}(undef, Int(floor((length(signal) - len)/step_size)), len)
    
    # update filter
    for k = 1:Int(floor((length(signal) - len - 1)/step_size))
        
        # update filter
        for i = 1:step_size
            Y = T*Y + u*signal[(k-1)*step_size + i]
        end
        
        # calculate FFT and save
        y[k,:] = FFTW.fft(Y[1:2:end])
        
    end
    
    return y
    
end

function reconstruct_warping(x::Array{Float64, 2}, z::Float64, len::Int, step_size::Int)
    # get update matrices
    T, u = allpass_update_matrix2(len, z)
    
    # specify output
    y = Float64[]
    
    # loop through blocks
    for ki = 1:size(x,1)
        
        # create vector Y
        Y = zeros(2*len)
        
        # set values of Y
        Y[1:2:end] = x[ki,:]
        Y[2:2:end] = x[ki,:]
        
        # collect output samples
        for k = 1:step_size
            
            # push last sample to output
            push!(y, Y[end-1])
            
            # update internal states to collect last 
            Y = T*Y + u*0
            
        end
    end    
    return y
end

function warp_ifft(x::Array{Complex{Float64},2}, z::Float64, step_size::Int)
    return reconstruct_warping(collect(real.(hcat([FFTW.ifft(x[k,:]) for k = 1:size(x,1)]...))'), z, size(x,2), step_size)
end

function fftcoefs2realimag(x::Array{Complex{Float64}, 2})
    z = conj.(reverse(x[:,Int(size(x,2)/2)+1:end], dims=2))
    return hcat(real.(z), imag.(z[:,1:end-1]))
end

function realimag2fftcoefs(x::Array{Float64, 2})
    z = x[:,1:Int((size(x, 2)+1)/2)] .+ 1im*(hcat(x[:,Int((size(x, 2)+1)/2)+1:end], zeros(size(x,1),1)))
    return hcat(zeros(size(z,1), 1) .+ 0*im, z[:,1:end-1], conj.(reverse(z, dims=2)))
end
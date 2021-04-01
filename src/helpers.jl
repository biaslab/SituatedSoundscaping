import DSP: hanning, rect
using DSP
export pad, squeeze, dB10toNum, numtodB10, add_dim, singlesided2twosided, reconstruct_signal, hanning, rect, softmax

"""
This function pad a symbol with an underscore and a number (with a certain number of digits)
"""
function pad(x::Symbol, y::Int64; nr_digits::Int64=3)

    # add underscore and pad with zeros
    return x*:_*Symbol(lpad(y, nr_digits,'0'))

end
function pad(x::Symbol, y::Int64, z::Int64; nr_digits::Int64=3)

    # add underscore and pad with zeros
    return pad(pad(x, y; nr_digits=nr_digits), z; nr_digits=nr_digits)

end

"""
This function squeezes an array and removes its singleton dimensions.
"""
function squeeze(A::AbstractArray{T,N}) where {T,N}

    # find singleton dimensions
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)

    A = dropdims(A; dims=singleton_dims)

    # return array with dropped dimensions
    return A

end

"""
Add an extra singleton dimension after the last dimension of the array.
"""
add_dim(x::Array) = reshape(x, (size(x)...,1))


"""
This function converts a number in decibel to a number on the real axis.
"""
function dB10toNum(a::Real)

    # perform conversion
    a = 10^(a/10)

    # return value
    return a::Float64

end


"""
This function converts a number to the corresponding number in decibel.
"""
function numtodB10(a::Real)

    # perform conversion
    a = 10*log10(a)

    # return value
    return a::Float64

end


"""
    normalize_sum(x), normalize_sum!(x)
    
    Function that normalizes an array of elements, such that their elements sum to 1.
    The function allows for inplace normalization.
"""
function normalize_sum!(x::Array{Float64,1}) 
    x ./= sum(x)
end
function normalize_sum(x::Array{Float64,1}) 
    x ./ sum(x)
end
function normalize_sum(x::Array{Int64,1}) 
    x ./ sum(x)
end


"""
    softmax(x), softmax!(x)
    
    Function that performs the softmax operation over a vector.
    The maximum is subtracted for numerical stability.
    The function allows for inplace operations.
"""
function softmax(x::Array{Float64,1})

    m = maximum(x)
    map((i) -> exp(i-m), x) |> normalize_sum

end
function softmax!(x::Array{Float64,1})

    x .-= maximum(x)
    x .= exp.(x)
    normalize_sum!(x)

end


"""
    logsumexp(x)
    
    Function that calculates the log(sum(exp(x)))
    The maximum is subtracted for numerical stability.
"""
function logsumexp(x::Array{Float64,1})

    # calculate mean
    m = maximum(x)

    # define addmax function
    addmax(y::Float64) = y + m

    # perform log-sum-exp trick
    mapreduce((i) -> exp(i - m), +, x) |> log |> addmax
    
end


"""
Function for calculating the lognormal distribution
"""
function lognormalpropto(x::Float64, μ::Float64, γ::Float64)
    ( log(γ) - γ*(x - μ)^2 ) / 2
end
function lognormalvarpropto(x::Float64, μ::Float64, σ2::Float64)
    ( -log(σ2) - 1/σ2*(x - μ)^2 ) / 2
end
function lognormalvarpropto(x::Array{Float64,2}, μ::Array{Float64,2}, σ2::Array{Float64,2}, ki::Int64, kii::Int64, kiii::Int64)
    ( -log(σ2[kiii,kii]) - 1/σ2[kiii,kii]*(x[kiii,ki] - μ[kiii,kii])^2 ) / 2
end
function lognormalvarpropto!(γ::Array{Float64,1}, x::Array{Float64,2}, μ::Array{Float64,2}, σ2::Array{Float64,2}, ki::Int64, kii::Int64, kiii::Int64)
    γ[kii:kii] .+= ( -log(σ2[kiii,kii]) - 1/σ2[kiii,kii]*(x[kiii,ki] - μ[kiii,kii])^2 ) / 2
end
function lognormal(x::AbstractArray{Float64,1}, μ::AbstractArray{Float64,1}, γ::AbstractArray{Float64,1})
    i = -0.5*log(2*pi)*length(x)
    @inbounds @simd for k = 1:length(x)
        i += lognormalpropto(x[k], μ[k], γ[k])
    end
    return i
end

function lognormal(x::Array{Float64,2}, μ::AbstractArray{Float64,1}, γ::AbstractArray{Float64,1})
    reduce(vcat, map((x) -> lognormal(x, μ, γ), eachcol(x)))::Array{Float64,1}
end
function lognormal(x::Array{Float64,2}, μ::Array{Float64,2}, γ::Array{Float64,2})
    reduce(hcat, map((μ,γ) -> lognormal(x, μ, γ), eachcol(μ),eachcol(γ)))
end


# function singlesided2twosided(x::Array{Complex{Float64}, 2})
#     if size(x,2)%2 == 1
#         return hcat(x[:,1:end], conj.(reverse(x[:,2:end-1], dims=2)))
#     else
        
#     end
# end

# function reconstruct_signal(X_sep::Array{Array{Complex{Float64},2},1}, window::Function)

#     # allocate array for speech signal
#     x_sep = Array{Array{Float64,1},1}(undef, length(X_sep))

#     # loop through separated signals
#     for k = 1:length(X_sep)
#         # get sizes
#         block_length = size(X_sep[k],2)
#         if window == rect
#             block_step = block_length
#         elseif window == hanning
#             block_step = block_length ÷ 2
#         end 

#         # initialize arrays with zeros
#         x_sep[k] = zeros(Int((prod(size(X_sep[k])) + block_length)/(block_length ÷ block_step)))
        
#         # overlap-add
#         for ki = 1:size(X_sep[k],1)
#             x_sep[k][(ki-1)*block_step+1:(ki-1)*block_step+block_length] = x_sep[k][(ki-1)*block_step+1:(ki-1)*block_step+block_length] + real.(DSP.ifft(X_sep[k][ki,:]))
#         end
#     end

#     return x_sep

# end

# reconstruct_signal(X_sep::Array{Complex{Float64},2}, window::Function) = reconstruct_signal([X_sep], window)[1]
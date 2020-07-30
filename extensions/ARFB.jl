using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
#import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate


"""
Description:

    An element-wise multiplication with unknown process noise.
    When modelling Fourier coefficients, this is also known as a probabilistic phase vocoder:

    f(y, x, θ, w) = 𝒩(y | θ x, inv(w))

Interfaces:
    1. y (output vector)
    2. x (input vector)
    3. θ (autoregression coefficients)
    4. w (precision matrix)

Construction:
    AutoregressiveFilterbank(out, θ, in, γ, id=:some_id)

"""

mutable struct AutoregressiveFilterbank <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function AutoregressiveFilterbank(y, x, θ, w; id=generateId(AutoregressiveFilterbank))
        
        # ensure that the input arguments are random variables
        @ensureVariables(y, x, θ, w) 
        
        # create new object
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:w] = self.interfaces[4] = associate!(Interface(self), w)
        
        # return object
        return self
    end
end

# add shortcut for calling the filter bank
slug(::Type{AutoregressiveFilterbank}) = "ARFB"




function ruleVariationalARFBOutNPPP(marg_y::Nothing, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    mθ = ForneyLab.unsafeMean(marg_θ)
    mx = ForneyLab.unsafeMean(marg_x)
    mw = ForneyLab.unsafeMean(marg_w)
                        
    # calculate new parameters
    my = mθ .* mx
    wy = mw

    # create variational message
    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=wy*my, w=wy)

end


function ruleVariationalARFBIn1PNPP(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::Nothing, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})
    
    # caluclate required means
    my = ForneyLab.unsafeMean(marg_y)
    mθ = ForneyLab.unsafeMean(marg_θ)
    mw = ForneyLab.unsafeMean(marg_w)

    # calculate required variances
    vθ = ForneyLab.unsafeCov(marg_θ)

    # calculate new parameters
    wx = (vθ' + mθ*mθ') .* mw
    # mx = inv(wx) * Diagonal(mθ) * mw * my
    xix = Diagonal(mθ) * mw' * my
    
    # create variational message
    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=xix, w=wx)

end


function ruleVariationalARFBIn2PPNP(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::Nothing, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    my = ForneyLab.unsafeMean(marg_y)
    mx = ForneyLab.unsafeMean(marg_x)
    mw = ForneyLab.unsafeMean(marg_w)

    # calculate required variances
    vx = ForneyLab.unsafeCov(marg_x)

    # calculate new parameters
    wθ = (vx' + mx*mx') .* mw
    xix = Diagonal(mx) * mw' * my

    # create variational message
    Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=xix, w=wθ)

end


function ruleVariationalARFBIn3PPPN(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_w::Nothing)

    # calculate required means
    my = ForneyLab.unsafeMean(marg_y)
    mx = ForneyLab.unsafeMean(marg_x)
    mθ = ForneyLab.unsafeMean(marg_θ)

    # calculate required variances
    vy = ForneyLab.unsafeCov(marg_y)
    vx = ForneyLab.unsafeCov(marg_x)
    vθ = ForneyLab.unsafeCov(marg_θ)

    # calculate new parameters
    v = vy + my*my' - (mθ .* mx)*my' - my*(mx .* mθ)' + Diagonal(mθ)*vx*Diagonal(mθ) + Diagonal(mx)*vθ*Diagonal(mx)  + (mθ .* mx)*(mθ .* mx)' + vθ.*vx
    nu = size(v,1) + 2 

    # create variational message
    Message(MatrixVariate, ForneyLab.Wishart, v=inv(v), nu=nu)

end

@naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARFBOutNPPP)

@naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARFBIn1PNPP)

@naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARFBIn2PPNP)

@naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                      :outbound_type => Message{ForneyLab.Wishart},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARFBIn3PPPN)


function AR_PSD(θ::Array{Float64,1}, ω::Float64, ρ::Float64, σ2::Float64)::Array{Float64,1}
    return σ2 ./ (1 .+ ρ^2 .- 2*ρ*cos.(θ.-ω))
end

function sum_AR_PSD(θ::Array{Float64,1}, ω::Array{Float64,1}, ρ::Array{Float64,1}, σ2::Array{Float64,1})::Array{Float64,1}
    return sum([AR_PSD(θ, ω[m], ρ[m], σ2[m]) for m = 1:length(ω)])
end

sigmoid(x) = 1 ./ (1 .+ exp.(-x))
dsigmoid(x) = sigmoid(x).*(1 .- sigmoid(x))

function logMSE(y::Array{Float64,2}, θ::Array{Float64,1}, ω::Array{Float64,1}, ρ::Array{Float64,1}, σ2::Array{Float64,1})::Float64
    
    # calculate number of AR components
    M = length(ω)
    
    # calculate number of data points
    N = size(y,2)
    
    # calculate MSE
    mse = mean( abs2.(y' .- log.(sum_AR_PSD(θ, ω, ρ, σ2))))
    
    # return MSE
    return mse
    
end

function ∇α_logMSE(y::Array{Float64,2}, θ::Array{Float64,1}, ω::Array{Float64,1}, ρ::Array{Float64,1}, σ2::Array{Float64,1})::Array{Float64}
        
    # calculate number of AR components
    M = length(ω)
    
    # calculate number of data points
    N = size(y,2)
    
    # calculate gradient
    grad = -2 / N * squeeze(sum((y' .- log.(sum_AR_PSD(θ, ω, ρ, σ2))), dims=2))'*(1 ./ sum_AR_PSD(θ, ω, ρ, σ2) .* hcat([AR_PSD(θ, ω[m], ρ[m], σ2[m]) for m=1:length(ω)]...))

    # return gradient
    return squeeze(grad)
end

function ∇β_logMSE(y::Array{Float64,2}, θ::Array{Float64,1}, ω::Array{Float64,1}, ρ::Array{Float64,1}, σ2::Array{Float64,1})::Array{Float64}
        
    # calculate number of AR components
    M = length(ω)
    
    # calculate number of data points\omega[1]
    N = size(y,2)
    
    # calculate gradient
    grad = - 2 / N * squeeze(sum((y' .- log.(sum_AR_PSD(θ, ω, ρ, σ2))), dims=2))'*(1 ./ sum_AR_PSD(θ, ω, ρ, σ2) .* ( σ2 ./ (1 .+ ρ.^2 .- 2*ρ.*cos.(θ'.-ω)).^2)' .* (2*ρ.*sin.(θ'.-ω))' .* (ω.*(1 .- ω/π))')
    
    # return gradient
    return squeeze(grad)
end

function ∇γ_logMSE(y::Array{Float64,2}, θ::Array{Float64,1}, ω::Array{Float64,1}, ρ::Array{Float64,1}, σ2::Array{Float64,1})::Array{Float64}
        
    # calculate number of AR components
    M = length(ω)
    
    # calculate number of data points
    N = size(y,2)
    
    # calculate gradient
    grad =  2 / N * squeeze(sum((y' .- log.(sum_AR_PSD(θ, ω, ρ, σ2))), dims=2))'*(1 ./ sum_AR_PSD(θ, ω, ρ, σ2) .* (σ2 ./ (1 .+ ρ.^2 .- 2*ρ.*cos.(θ'.-ω)).^2)' .* (2*ρ .-2*cos.(θ'.-ω))' .* (ρ .*(1 .- ρ ))')
    
    # return gradient
    return squeeze(grad)
end
function Adam(θ::Array{Float64,1}, ∇::Array{Float64,1}, s::Array{Float64,1}, r::Array{Float64,1}, it::Int64; η=0.001::Float64, ρ1=0.9::Float64, ρ2=0.999::Float64, δ=1e-7::Float64)
    
    # update first moment
    s = ρ1*s + (1-ρ1)*∇
    
    # update second moment
    r = ρ2*r + (1-ρ2)*∇.*∇
    
    # perform bias correction
    sx = s / (1 - ρ1^it)
    rx = r / (1 - ρ2^it)
    
    # update parameters
    θ = θ - η./(δ .+ sqrt.(rx)) .* sx
    
    return θ, s, r
end
;

function pretrain_ARFB(yi, nr_freqs)
    
    freqres = 100
    smooth = 1000
    psd = twosided2singlesided(PSDovertime(yi, freqres*2, freqres*2-1, rectangularwindow))[:,:]
    log_psd_smooth = vcat([log.(mean(psd[k:k+smooth,:], dims =1)) for k = 1:size(psd,1)-smooth]...)[:,:]
    ;
    
    L = size(log_psd_smooth, 2) # length of data
    M = nr_freqs # nr_clusters
    N = size(log_psd_smooth, 1) # data points

    # initializers
    θ = collect(0:π/L:π - π/L)
    ρ = rand(M)/100 .+0.99 # in range 0.9-1.0
    σ2 = 10*rand(M)
    α = log.(σ2)
    ω = collect(π/(M+1):π/(M+1):π-π/(M+1)+0.0001)
    β = -log.(π./ω .- 1)
    γ = -log.(1 ./ ρ .-1)

    # initializers:
    s_α = zeros(M)
    r_α = zeros(M)
    s_β = zeros(M)
    r_β = zeros(M)
    s_γ = zeros(M)
    r_γ = zeros(M)

    # optimization
    nits = 5000
    η1 = 0.001
    η2 = 0.00001
    η3 = 0.00001

    batches = 50
    step = Int(floor(size(log_psd_smooth,1)/batches))

    p = Progress(nits)
    
    for it = 1:nits
        for b = 1:batches
            α, s_α, r_α = Adam(α, ∇α_logMSE(log_psd_smooth[(b-1)*step+1:b*step, :], θ, ω, ρ, σ2), s_α, r_α, it, η=η1)
            σ2 = exp.(α)
            β, s_β, r_β = Adam(β, ∇β_logMSE(log_psd_smooth[(b-1)*step+1:b*step, :], θ, ω, ρ, σ2), s_β, r_β, it, η=η2)
            ω = π*sigmoid.(β)
            γ, s_γ, r_γ = Adam(γ, ∇γ_logMSE(log_psd_smooth[(b-1)*step+1:b*step, :], θ, ω, ρ, σ2), s_γ, r_γ, it, η=η3)
            ρ = sigmoid(γ)
        end
        η1 = η1*0.99999
        η2 = η2*0.99999
        η3 = η3*0.99999
        mse = logMSE(log_psd_smooth, θ, ω, ρ, σ2)
        
        next!(p)
        
    end
    
    return  ω, ρ, σ2
    
end
using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!, matches
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate, unsafeMeanVector

"""
Description:

    This node implements a Gaussian scale mixture model

    f(X, z, ξ...) = Π_k 𝒩𝒞(X | μ=0, Γ=exp(ξ), C=0)^{z_k}

Interfaces:
    1. X (complex output vector, complex Fourier coefficients)
    2. z (categorical input vector, cluster switch)
    3+. ξ... (real input vector, real "log-power" spectrum)

Construction:
    GSMM(X, z, ξ..., id=:some_id)

"""

mutable struct GaussianScaleMixtureModel <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GaussianScaleMixtureModel(X, z, ξ::Vararg; id=generateId(GaussianScaleMixtureModel))
        
        # ensure that the input arguments are random variables
        @ensureVariables(X, z) 
        for k = 1:length(ξ)
            @ensureVariables(ξ[k])
        end
        
        # create new object
        self = new(id, Array{Interface}(undef, 2+length(ξ)), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:X] = self.interfaces[1] = associate!(Interface(self), X)
        self.i[:z] = self.interfaces[2] = associate!(Interface(self), z)
        for k=1:length(ξ)
            self.i[:X*k] = self.interfaces[2+k] = associate!(Interface(self), ξ[k])
        end
        
        # return object
        return self
    end
end

slug(::Type{GaussianScaleMixtureModel}) = "GSMM"

function ruleVBGaussianScaleMixtureModelΞ(marg_X::ProbabilityDistribution{Multivariate},
                                          marg_z::ProbabilityDistribution{Univariate},
                                          marg_ξ::Vararg{Union{Nothing, ProbabilityDistribution{Multivariate}}})
    # Find message to send
    k = findfirst(marg_ξ .== nothing) 
    
    # get expectation of z
    p = clamp.(unsafeMeanVector(marg_z), tiny, 1.0 - tiny)

    # calculate mean and precision
    m = log.(diag(unsafeCov(marg_X)) + abs2.(unsafeMean(marg_X)))
    w = p[k]*Ic(length(m))
    
    return Message(Multivariate, GaussianMeanPrecision, m=m, w=w)
end

function ruleVBGaussianScaleMixtureModelZ(marg_X::ProbabilityDistribution{Multivariate},
                                          marg_z::Nothing,
                                          marg_ξ::Vararg{ProbabilityDistribution{Multivariate}})

    # calculate variables of X
    m_X = unsafeMean(marg_X)
    v_X = diag(unsafeCov(marg_X))
    
    # calculate "average energies"
    U = Vector{Float64}(undef, length(marg_ξ))
    for k = 1:length(marg_ξ)
        m_ξ = unsafeMean(marg_ξ[k])
        v_ξ = diag(unsafeCov(marg_ξ[k]))
        U[k] = sum(-m_ξ - exp.(-m_ξ + v_ξ/2).*(v_ξ + abs2.(m_ξ)))
    end

    return Message(Univariate, Categorical, p=ForneyLab.softmax(-U))
end

function ruleVBGaussianScaleMixtureModelOut(marg_X::Nothing,
                                  marg_z::ProbabilityDistribution{Univariate},
                                  marg_ξ::Vararg{ProbabilityDistribution{Multivariate}})
    # get class probabilities
    p = clamp.(unsafeMeanVector(marg_z), tiny, 1.0 - tiny)

    # get mean and variance
    w = 0
    for k = 1:length(p)
        w = w .+ p[k]*exp.(-unsafeMean(marg_ξ[k]) + diag(unsafeCov(marg_ξ[k]))/2)
    end
    m = zeros(length(w)) .+ 0.0im
        
    return Message(Multivariate, ComplexNormal, μ=m, Γ=diagm(1 ./ w).+0.0im, C=mat(0.0+0.0im))
end    



mutable struct VBGaussianScaleMixtureModelZ <: NaiveVariationalRule{GaussianScaleMixtureModel} end
ForneyLab.outboundType(::Type{VBGaussianScaleMixtureModelZ}) = Message{Categorical}
function ForneyLab.isApplicable(::Type{VBGaussianScaleMixtureModelZ}, input_types::Vector{<:Type})
    (length(input_types) > 2) || return false
    for (i, input_type) in enumerate(input_types)
        if (i == 2)
            (input_type == Nothing) || return false
        else
            matches(input_type, ProbabilityDistribution) || return false
        end
    end
    return true
end

function matchPVInputs(input_types::Vector{<:Type})
    Nothing_positions = []
    p_positions = []
    for (i, input_type) in enumerate(input_types)
        if matches(input_type, ProbabilityDistribution)
            push!(p_positions, i)
        elseif (input_type == Nothing)
            push!(Nothing_positions, i)
        end
    end

    return (Nothing_positions, p_positions)
end

mutable struct VBGaussianScaleMixtureModelΞ <: NaiveVariationalRule{GaussianScaleMixtureModel} end
ForneyLab.outboundType(::Type{VBGaussianScaleMixtureModelΞ}) = Message{GaussianMeanPrecision}
function ForneyLab.isApplicable(::Type{VBGaussianScaleMixtureModelΞ}, input_types::Vector{<:Type})
    n_inputs = length(input_types)

    (Nothing_positions, p_positions) = matchPVInputs(input_types)
    n_Nothings = length(Nothing_positions)
    n_ps = length(p_positions)

    (n_Nothings == 1) || return false
    (n_Nothings + n_ps == n_inputs) || return false
    (1 in p_positions) || return false
    (2 in p_positions) || return false

    return true
end

mutable struct VBGaussianScaleMixtureModelOut <: NaiveVariationalRule{GaussianScaleMixtureModel} end
ForneyLab.outboundType(::Type{VBGaussianScaleMixtureModelOut}) = Message{ComplexNormal}
function ForneyLab.isApplicable(::Type{VBGaussianScaleMixtureModelOut}, input_types::Vector{<:Type})
    for (i, input_type) in enumerate(input_types)
        if (i == 1)
            (input_type == Nothing) || return false
        else
            matches(input_type, ProbabilityDistribution) || return false
        end
    end
    return true
end
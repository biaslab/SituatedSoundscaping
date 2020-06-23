using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate

"""
Description:

    This node implements a complex normal distribution whose precision is a function of its input variables.

    f(X, ξ) = 𝒩𝒞(X | μ=0, Γ=exp(ξ), C=0)

Interfaces:
    1. X (complex output vector, complex Fourier coefficients)
    2. ξ (real input vector, real "log-power" spectrum)

Construction:
    ComplexHGF(X, ξ, id=:some_id)

"""

mutable struct ComplexHGF <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function ComplexHGF(X, ξ; id=generateId(HGF))
        
        # ensure that the input arguments are random variables
        @ensureVariables(X, ξ) 
        
        # create new object
        self = new(id, Array{Interface}(undef, 2), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:X] = self.interfaces[1] = associate!(Interface(self), X)
        self.i[:ξ] = self.interfaces[2] = associate!(Interface(self), ξ)
        
        # return object
        return self
    end
end

slug(::Type{ComplexHGF}) = "CHGF"



function ruleVariationalComplexHGFOutNP(marg_X::Nothing, 
                                 marg_ξ::ProbabilityDistribution{Multivariate})
    
    # caluclate required mean
    mξ = unsafeMean(marg_ξ)

    # calculate required variance
    vξ = diag(unsafeCov(marg_ξ))

    # calculate new parameters
    mX = zeros(size(mξ)) .+ 0im
    vX = exp.(mξ - vξ/2) .+ 0im
    
    # create variational message
    return Message(Multivariate, ComplexNormal, μ=mX, Γ=diagm(vX), C=mat(0.0+0.0im))

end


function ruleVariationalComplexHGFIn1PN(marg_X::ProbabilityDistribution{Multivariate}, 
                                 marg_ξ::Nothing)
    
    # calculate required means
    mX = unsafeMean(marg_X)

    # calculate required variances
    vX = diag(unsafeCov(marg_X))

    # calculate new parameters
    mξ = log.(abs2.(mX) + real.(vX))
    vξ = 1.0*ones(length(mξ))

    # create variational message
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=mξ./vξ, w=diagm(1 ./ vξ))

end

@naiveVariationalRule(:node_type     => ComplexHGF,
                      :outbound_type => Message{ComplexNormal},
                      :inbound_types => (Nothing, ProbabilityDistribution),
                      :name          => VariationalComplexHGFOutNP)

@naiveVariationalRule(:node_type     => ComplexHGF,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing),
                      :name          => VariationalComplexHGFIn1PN)
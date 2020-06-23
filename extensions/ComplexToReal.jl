using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate
     


"""
Description:

    Converts an array of complex numbers into a concatenated array of their real and imaginary     parts.
    Note: in the derivations, C=0 is assumed for simplicity, at the cost of generality.

    f(rx, cx) = vcat(real.(cx), imag.(cx))

Interfaces:
    1. rx (real output vector)
    2. cx (complex input vector)

Construction:
    ComplexNormal(rx, cx, id=:some_id)

"""

mutable struct ComplexToReal <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function ComplexToReal(rx, cx; id=generateId(ComplexToReal))
        
        # ensure that the input arguments are random variables
        @ensureVariables(rx, cx) 
        
        # create new object
        self = new(id, Array{Interface}(undef, 2), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:rx] = self.interfaces[1] = associate!(Interface(self), rx)
        self.i[:cx] = self.interfaces[2] = associate!(Interface(self), cx)
        
        # return object
        return self
    end
end

slug(::Type{ComplexToReal}) = "C2R"


function ruleSPComplexToRealOutNC(marg_rx::Nothing, 
                                  marg_cx::Message{ComplexNormal})

    # calculate mean of complex random variable
    μ_cx = unsafeMean(marg_cx.dist)
    
    # calculate covariance of complex random variable
    Γ_cx = unsafeCov(marg_cx.dist)
    
    # convert the mean vector μ_cx to the mean vector [real(μ_cx), imag(μ_cx)]
    μ_rx = vcat(real.(μ_cx), imag.(μ_cx))
    
    # calculate precision matrix
    w_rx = vcat(hcat(0.5*inv(real(Γ_cx)), zeros(size(Γ_cx))), hcat(zeros(size(Γ_cx)), 0.5*inv(real(Γ_cx))))
        
    # create sum-product message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=w_rx*μ_rx, w=w_rx)


end


function ruleSPComplexToRealIn1GN(marg_rx::Message{GaussianWeightedMeanPrecision},
                                  marg_cx::Nothing)

    # calculate mean of random variable
    μ_rx = unsafeMean(marg_rx.dist)
    
    # calculate covariance of random variable
    Σ_rx = unsafeCov(marg_rx.dist)
    
    # calculate complex vector length
    N = Int(round(length(μ_rx)/2))
    
    # calculate mean vector of complex random variable
    μ_cx = μ_rx[1:N] + 1im*μ_rx[N+1:end]
    
    # calculate complex covariance matrix of complex random variable
    V_xx = Σ_rx[1:N, 1:N]
    V_yy = Σ_rx[N+1:end, N+1:end]
    V_xy = Σ_rx[1:N, N+1:end]
    V_yx = Σ_rx[N+1:end, 1:N]
    Γ_cx = V_xx + V_yy .+ 1im*(V_yx-V_xy)
    
    # create sum-product message
    return Message(Multivariate, ComplexNormal, μ=μ_cx, Γ=Γ_cx, C=mat(0.0+0.0im))

end

@sumProductRule(:node_type     => ComplexToReal,
                :outbound_type => Message{GaussianWeightedMeanPrecision},
                :inbound_types => (Nothing, Message{ComplexNormal}),
                :name          => SPComplexToRealOutNC)

@sumProductRule(:node_type     => ComplexToReal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Message{GaussianWeightedMeanPrecision}, Nothing),
                :name          => SPComplexToRealIn1GN)
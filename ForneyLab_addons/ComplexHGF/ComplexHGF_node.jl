using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
                    

export ComplexHGF, CHGF, slug

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
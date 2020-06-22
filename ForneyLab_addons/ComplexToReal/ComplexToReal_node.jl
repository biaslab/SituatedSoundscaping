using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
                    

export ComplexToReal, C2R, slug

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
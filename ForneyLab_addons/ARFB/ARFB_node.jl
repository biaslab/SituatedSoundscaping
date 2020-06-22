using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!

export AutoregressiveFilterbank, ARFB, slug

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
using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate


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
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    mθ = unsafeMean(marg_θ)
    mx = unsafeMean(marg_x)
    mw = unsafeMean(marg_w)
                        
    # calculate new parameters
    my = mθ .* mx
    wy = mw

    # create variational message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=wy*my, w=wy)

end


function ruleVariationalARFBIn1PNPP(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::Nothing, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})
    
    # caluclate required means
    my = unsafeMean(marg_y)
    mθ = unsafeMean(marg_θ)
    mw = unsafeMean(marg_w)

    # calculate required variances
    vθ = unsafeCov(marg_θ)

    # calculate new parameters
    wx = (vθ' + mθ*mθ') .* mw
    mx = inv(wx) * Diagonal(mθ) * mw * my

    # create variational message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=wx*mx, w=wx)

end


function ruleVariationalARFBIn2PPNP(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::Nothing, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mw = unsafeMean(marg_w)

    # calculate required variances
    vx = unsafeCov(marg_x)

    # calculate new parameters
    wθ = (vx' + mx*mx') .* mw
    mθ = inv(wθ) * Diagonal(mx) * mw * my

    # create variational message
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=wθ*mθ, w=wθ)

end


function ruleVariationalARFBIn3PPPN(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::Nothing)

    # calculate required means
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mθ = unsafeMean(marg_θ)

    # calculate required variances
    vy = unsafeCov(marg_y)
    vx = unsafeCov(marg_x)
    vθ = unsafeCov(marg_θ)

    # calculate new parameters
    v = vy + my*my' - (mθ .* mx)*my' - my*(mx .* mθ)' + Diagonal(mθ)*vx*Diagonal(mθ) + Diagonal(mx)*vθ*Diagonal(mx)  + (mθ .* mx)*(mθ .* mx)' + vθ.*vx
    nu = size(v,1) + 2 

    # create variational message
    Message(MatrixVariate, Wishart, v=inv(v), nu=nu)

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
                      :outbound_type => Message{Wishart},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARFBIn3PPPN)

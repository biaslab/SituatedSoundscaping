using ForneyLab

mutable struct HGF <: ForneyLab.SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function HGF(X, ξ; id=ForneyLab.generateId(HGF))
        
        # ensure that the input arguments are random variables
        @ensureVariables(X, ξ) 
        
        # create new object
        self = new(id, Array{Interface}(undef, 2), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        ForneyLab.addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:X] = self.interfaces[1] = ForneyLab.associate!(Interface(self), X)
        self.i[:ξ] = self.interfaces[2] = ForneyLab.associate!(Interface(self), ξ)
        
        # return object
        return self
    end
end

function ruleVariationalHGFOutNP(marg_X::Nothing, 
                                 marg_ξ::ProbabilityDistribution{ForneyLab.Multivariate})
    
    # caluclate required mean
    mξ = ForneyLab.unsafeMean(marg_ξ)

    # calculate required variance
    vξ = diag(ForneyLab.unsafeCov(marg_ξ))

    # calculate new parameters
    mX = zeros(size(mξ)) .+ 0im
    vX = exp.(mξ - vξ/2) .+ 0im
    
    # create variational message
    return Message(ForneyLab.Multivariate, ComplexNormal, μ=mX, Γ=diagm(vX), C=mat(0.0+0.0im))

end

function ruleVariationalHGFIn1PN(marg_X::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                 marg_ξ::Nothing)
    
    # calculate required means
    mX = ForneyLab.unsafeMean(marg_X)

    # calculate required variances
    vX = diag(ForneyLab.unsafeCov(marg_X))

    # calculate new parameters
    mξ = log.(abs2.(mX) + real.(vX))
    vξ = 1.0*ones(length(mξ))

    # create variational message
    Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=mξ./vξ, w=diagm(1 ./ vξ))

end

@naiveVariationalRule(:node_type     => HGF,
                      :outbound_type => Message{ComplexNormal},
                      :inbound_types => (Nothing, ProbabilityDistribution),
                      :name          => VariationalHGFOutNP)

@naiveVariationalRule(:node_type     => HGF,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing),
                      :name          => VariationalHGFIn1PN)
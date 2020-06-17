mutable struct AutoregressiveFilterbank <: ForneyLab.SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function AutoregressiveFilterbank(y, x, θ, γ; id=ForneyLab.generateId(AutoregressiveFilterbank))
        
        # ensure that the input arguments are random variables
        @ensureVariables(y, x, θ, γ) 
        
        # create new object
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        ForneyLab.addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:y] = self.interfaces[1] = ForneyLab.associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = ForneyLab.associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = ForneyLab.associate!(Interface(self), θ)
        self.i[:γ] = self.interfaces[4] = ForneyLab.associate!(Interface(self), γ)
        
        # return object
        return self
    end
end

# add shortcut for calling the filter bank
slug(::Type{AutoregressiveFilterbank}) = "ARFB"

# create random variable for vectors of gamma distributed RV's
mutable struct GammaVector <: ForneyLab.SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GammaVector(out, a, b; id=ForneyLab.generateId(GammaVector))
        @ensureVariables(out, a, b)
        self = new(id, Array{Interface}(undef, 3), Dict{Symbol,Interface}())
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = ForneyLab.associate!(Interface(self), out)
        self.i[:a] = self.interfaces[2] = ForneyLab.associate!(Interface(self), a)
        self.i[:b] = self.interfaces[3] = ForneyLab.associate!(Interface(self), b)

        return self
    end
end

format(dist::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}) = "$(slug(GammaVector))(a=$(format(dist.params[:a])), b=$(format(dist.params[:b])))"

ProbabilityDistribution(::Type{ForneyLab.Multivariate}, ::Type{GammaVector}; a::Array{Float64,1}, b::Array{Float64,1}) = ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}(Dict(:a=>a, :b=>b))
ProbabilityDistribution(::Type{GammaVector}; a::Array{Float64,1}, b::Array{Float64,1}) = ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}(Dict(:a=>a, :b=>b))

function ForneyLab.prod!( x::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector},
                y::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector},
                z::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}=ProbabilityDistribution(ForneyLab.Multivariate, GammaVector, a=ones(size(x.params[:a])), b=ones(size(x.params[:a]))))

    z.params[:a] = x.params[:a] + y.params[:a] .- 1.0
    z.params[:b] = x.params[:b] + y.params[:b]

    return z
end

function ForneyLab.sample(dist::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector})
    a, b = dist.params[:a], dist.params[:b]
    return [ForneyLab.sample(ProbabilityDistribution(ForneyLab.Univariate, ForneyLab.Gamma, a=a[k], b=b[k])) for k = 1:length(a)]
end

unsafeMean(dist::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}) = dist.params[:a] ./ dist.params[:b] # unsafe mean  
ForneyLab.unsafeMean(dist::ProbabilityDistribution{ForneyLab.Multivariate, GammaVector}) = dist.params[:a] ./ dist.params[:b] # unsafe mean  


function ruleVariationalARFBOutNPPP(marg_y::Nothing, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_γ::ProbabilityDistribution{ForneyLab.Multivariate})

    # calculate required means
    mθ = ForneyLab.unsafeMean(marg_θ)
    mx = ForneyLab.unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)
                        
    # calculate new parameters
    my = mθ .* mx
    vy = 1 ./ mγ

    # create variational message
    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=my ./ vy, w=diagm(1 ./ vy))

end

function ruleVariationalARFBIn1PNPP(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::Nothing, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_γ::ProbabilityDistribution{ForneyLab.Multivariate})
    
    # caluclate required means
    my = ForneyLab.unsafeMean(marg_y)
    mθ = ForneyLab.unsafeMean(marg_θ)
    mγ = ForneyLab.unsafeMean(marg_γ)

    # calculate required variances
    vθ = diag(ForneyLab.unsafeCov(marg_θ))

    # calculate new parameters
    mx = mθ.*my./(vθ + mθ.^2)
    vx = 1 ./(mγ.*(vθ + mθ.^2))

    # create variational message
    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=mx./vx, w=diagm(1 ./vx))

end

function ruleVariationalARFBIn2PPNP(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::Nothing, 
                                    marg_γ::ProbabilityDistribution{ForneyLab.Multivariate})

    # calculate required means
    my = ForneyLab.unsafeMean(marg_y)
    mx = ForneyLab.unsafeMean(marg_x)
    mγ = unsafeMean(marg_γ)

    # calculate required variances
    vx = diag(ForneyLab.unsafeCov(marg_x))

    # calculate new parameters
    mθ = mx.*my./(vx + mx.^2)
    vθ = 1 ./(mγ.*(vx + mx.^2))

    # create variational message
    Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=mθ./vθ, w=diagm(1 ./vθ))

end

function ruleVariationalARFBIn3PPPN(marg_y::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_x::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_θ::ProbabilityDistribution{ForneyLab.Multivariate}, 
                                    marg_γ::Nothing)

    # calculate required means
    my = ForneyLab.unsafeMean(marg_y)
    mx = ForneyLab.unsafeMean(marg_x)
    mθ = ForneyLab.unsafeMean(marg_θ)

    # calculate required variances
    vy = diag(ForneyLab.unsafeCov(marg_y))
    vx = diag(ForneyLab.unsafeCov(marg_x))
    vθ = diag(ForneyLab.unsafeCov(marg_θ))

    # calculate new parameters
    b = 1/2*((vy + my.^2) + (vθ + mθ.^2).*(vx + mx.^2) - 2*mθ.*mx.*my)
    a = 3/2*ones(size(b))

    # create variational message
    Message(ForneyLab.Multivariate, GammaVector, a=a, b=b)

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
                      :outbound_type => Message{GammaVector},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARFBIn3PPPN)



function generateARFB(nr_γ, bufsize)
    
    model = quote
        
        fg = FactorGraph()
        
        @RV [id=:smin] smin ~ GaussianMeanVariance(placeholder(:μ_smin, dims=($nr_γ,)), placeholder(:Σ_smin, dims=($nr_γ,$nr_γ)))
        @RV [id=:θ] θ ~ GaussianMeanVariance(placeholder(:μ_θ, dims=($nr_γ,)), placeholder(:Σ_θ, dims=($nr_γ,$nr_γ)))
        @RV [id=:γ] γ ~ GammaVector(placeholder(:a_γ, dims=($nr_γ,)), placeholder(:b_γ, dims=($nr_γ,)))
        @RV [id=:s] s ~ AutoregressiveFilterbank(smin, θ, γ)
        @RV [id=:x] x = placeholder(:c, dims=($bufsize,$nr_γ)) * s
        @RV [id=:y] y ~ GaussianMeanVariance(x, placeholder(:Σ_x, dims=($bufsize,$bufsize)))
        placeholder(y, :y, dims=($bufsize,))
        
        q = PosteriorFactorization(smin, s, θ, γ, ids=[:smin, :s, :θ, :γ])

    end
    
end

ruleSPGammaVectorOutNPP(msg_out::Nothing, 
                        msg_a::Message{PointMass},
                        msg_b::Message{PointMass}) =
                        Message(ForneyLab.Multivariate, GammaVector, a=deepcopy(msg_a.dist.params[:m]), b=deepcopy(msg_b.dist.params[:m]))

ruleVBGammaVectorOut(dist_out::Any,
                     dist_a::ProbabilityDistribution{ForneyLab.Multivariate},
                     dist_b::ProbabilityDistribution{ForneyLab.Multivariate}) =
                     Message(ForneyLab.Multivariate, GammaVector, a=ForneyLab.unsafeMean(dist_a), b=ForneyLab.unsafeMean(dist_b))

@sumProductRule(:node_type     => GammaVector,
                :outbound_type => Message{GammaVector},
                :inbound_types => (Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPGammaVectorOutNPP)

@naiveVariationalRule(:node_type     => GammaVector,
                      :outbound_type => Message{GammaVector},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VBGammaVectorOut)


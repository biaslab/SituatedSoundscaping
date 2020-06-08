mutable struct AutoregressiveFilterbank <: ForneyLab.SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function AutoregressiveFilterbank(y, x, θ, γ...; id=ForneyLab.generateId(AutoregressiveFilterbank))
        
        # ensure that the input arguments are random variables
        #@ensureVariables(y, x, θ, γ...) # proves to be difficult with the γ argument
        
        # create new object
        self = new(id, Array{Interface}(undef, 3 + length(γ)), Dict{Symbol,Interface}())
        
        # add the node to the current factor graph
        ForneyLab.addNode!(currentGraph(), self)
        
        # add argument variables to interfaces of node
        self.i[:y] = self.interfaces[1] = ForneyLab.associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = ForneyLab.associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = ForneyLab.associate!(Interface(self), θ)
        for k = 1:length(γ) # add variable amount of γ arguments
            self.i[pad(:γ,k)] = self.interfaces[3+k] = ForneyLab.associate!(Interface(self), γ[k])
        end
        
        # return object
        return self
    end
end

# add shortcut for calling the filter bank
slug(::Type{AutoregressiveFilterbank}) = "ARFB"


# This function creates the function name of the update rules that corresponds to the filter bank with nr_γ frequency bins for the message at the k'th interface
function functionnames(nr_γ, k)
    
    # if k =1, then the name should include "Out"
    if k == 1

        name = "ruleVariationalARFBMVOutN"
        name = name*"P"^(nr_γ+2)
        
    # if k > 1, then the name should include "In"
    else
        
        name = "ruleVariationalARFBMVIn"*string(k-1)
        # add temporary string for the correct positioning of the N
        tmp = collect("P"^(nr_γ+3))
        tmp[k] = 'N'
        tmp = String(tmp)
        name = name*tmp
        
    end
    
    # return function name
    return name

end

# this function creates union types for the update rule macros
function generateunionstypes()
    uniontypes = quote
        MVprobnothing = Union{ProbabilityDistribution{ForneyLab.Multivariate},Nothing}
        UVprobnothing = Union{ProbabilityDistribution{ForneyLab.Univariate},Nothing}
    end
    return uniontypes
end

# this function generates macros for the update rules of an ARFB with nr_γ variances
function generateupdaterules(nr_γ)
    
    # create array to store macro expressions
    functions = Array{Expr}(undef, nr_γ+3)
    
    # loop through interfaces
    for k = 1:nr_γ+3
        
        # specify message for y
        if k == 1
            tmp = Symbol(functionnames(nr_γ,k))
            functions[k] = quote
                function $tmp(marg_y::MVprobnothing, marg_x::MVprobnothing, marg_θ::MVprobnothing, marg_γ...)
                    
                    mθ = ForneyLab.unsafeMean(marg_θ)
                    mx = ForneyLab.unsafeMean(marg_x)
                    mγ = Array{Float64,1}(undef, length(mθ))
                    for i = 1:length(mγ)
                        mγ[i] = ForneyLab.unsafeMean(marg_γ[i])
                    end
                    
                    # calculate new parameters
                    my = mθ .* mx
                    vy = 1 ./ mγ

                    # create variational message
                    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=my ./ vy, w=diagm(1 ./ vy))
                
                end
            end
            
        # specify message for x
        elseif k == 2
            tmp = Symbol(functionnames(nr_γ,k))
            functions[k] = quote
                function $tmp(marg_y::MVprobnothing, marg_x::MVprobnothing, marg_θ::MVprobnothing, marg_γ...)
                    
                    # calculate required means
                    my = ForneyLab.unsafeMean(marg_y)
                    mθ = ForneyLab.unsafeMean(marg_θ)
                    mγ = Array{Float64,1}(undef, length(mθ))
                    for i = 1:length(mγ)
                        mγ[i] = ForneyLab.unsafeMean(marg_γ[i])
                    end

                    # calculate required variances
                    vθ = diag(ForneyLab.unsafeCov(marg_θ))

                    # calculate new parameters
                    mx = mθ.*my./(vθ + mθ.^2)
                    vx = 1 ./(mγ.*(vθ + mθ.^2))

                    # create variational message
                    return Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=mx./vx, w=diagm(1 ./vx))
                
                end
            end 
        
        # specify message for θ
        elseif k == 3
            tmp = Symbol(functionnames(nr_γ,k))
            functions[k] = quote
                function $tmp(marg_y::MVprobnothing, marg_x::MVprobnothing, marg_θ::MVprobnothing, marg_γ...)
                    
                    # calculate required means
                    my = ForneyLab.unsafeMean(marg_y)
                    mx = ForneyLab.unsafeMean(marg_x)
                    mγ = Array{Float64,1}(undef, length(my))
                    for i = 1:length(mγ)
                        mγ[i] = ForneyLab.unsafeMean(marg_γ[i])
                    end

                    # calculate required variances
                    vx = diag(ForneyLab.unsafeCov(marg_x))

                    # calculate new parameters
                    mθ = mx.*my./(vx + mx.^2)
                    vθ = 1 ./(mγ.*(vx + mx.^2))

                    # create variational message
                    Message(ForneyLab.Multivariate, GaussianWeightedMeanPrecision, xi=mθ./vθ, w=diagm(1 ./vθ))
                
                end
            end
            
        # specify messages for γ
        else
            tmp = Symbol(functionnames(nr_γ,k))
            functions[k] = quote
                function $tmp(marg_y::MVprobnothing, marg_x::MVprobnothing, marg_θ::MVprobnothing, marg_γ...)
                    
                    # calculate required means
                    my = ForneyLab.unsafeMean(marg_y)
                    mx = ForneyLab.unsafeMean(marg_x)
                    mθ = ForneyLab.unsafeMean(marg_θ)

                    # calculate required variances
                    vy = diag(ForneyLab.unsafeCov(marg_y))
                    vx = diag(ForneyLab.unsafeCov(marg_x))
                    vθ = diag(ForneyLab.unsafeCov(marg_θ))

                    # calculate new parameters
                    a = 3/2
                    b = 1/2*((vy + my.^2) + (vθ + mθ.^2).*(vx + mx.^2) - 2*mθ.*mx.*my)[$k-3]

                    # create variational message
                    Message(ForneyLab.Gamma, a=a, b=b)
                
                end
            end
            
         end
        
    end
    
    # return update rule macros
    return functions
    
end

# this function specifies the update rules functions as ForneyLab messages
function generateFLupdaterules(nr_γ)
    
    updaterules = Array{Expr}(undef, nr_γ+3)
    
    for k = 1:nr_γ+3
        
        if k < 4
            tmp1 = "ProbabilityDistribution, "^(k-1)*"Nothing"*", ProbabilityDistribution"^(nr_γ+3-k)
            tmp2 = Symbol(functionnames(nr_γ,k)[5:end])
            updaterules[k] = quote
                
                @naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                                      :inbound_types => $(Meta.parse(tmp1)),
                                      :name          => $tmp2)
                
            end
            
         else
            tmp1 = "ProbabilityDistribution, "^(k-1)*"Nothing"*", ProbabilityDistribution"^(nr_γ+3-k)
            tmp2 = Symbol(functionnames(nr_γ,k)[5:end])
            updaterules[k] = quote
                
                @naiveVariationalRule(:node_type     => AutoregressiveFilterbank,
                                      :outbound_type => Message{ForneyLab.Gamma},
                                      :inbound_types => $(Meta.parse(tmp1)),
                                      :name          => $tmp2)
                
            end
        end
                
    end
    
    return updaterules
    
end

# This function loads the update rules, union types and such
function prepareARFB(nr_γ)
    
    # create union types
    eval(generateunionstypes())
    
    # create update rules
    updaterules = generateupdaterules(nr_γ)
    for k = 1:length(updaterules)
        eval(updaterules[k])
    end
    
    # specify the variational rules to FL
    FLrules = generateFLupdaterules(nr_γ)
    for k = 1:length(FLrules)
        eval(FLrules[k])
    end
    
end

function generateARFB(nr_γ, bufsize)
    
    model = quote
        
        fg = FactorGraph()
        
        @RV [id=:smin] smin ~ GaussianMeanVariance(placeholder(:μ_smin, dims=($nr_γ,)), placeholder(:Σ_smin, dims=($nr_γ,$nr_γ)))
        @RV [id=:θ] θ ~ GaussianMeanVariance(placeholder(:μ_θ, dims=($nr_γ,)), placeholder(:Σ_θ, dims=($nr_γ,$nr_γ)))
        γ = Array{Variable,1}(undef, $nr_γ)
        for k = 1:$nr_γ
            @RV [id=pad(:γ,k)] γ[k] ~ ForneyLab.Gamma(placeholder(pad(:a_γ, k)), placeholder(pad(:b_γ,k)))
        end
        @RV [id=:s] s ~ AutoregressiveFilterbank(smin, θ, γ...)
        @RV [id=:x] x = placeholder(:c, dims=($bufsize,$nr_γ)) * s
        @RV [id=:y] y ~ GaussianMeanVariance(x, placeholder(:Σ_x, dims=($bufsize,$bufsize)))
        placeholder(y, :y, dims=($bufsize,))
        
        ids = vcat([:smin, :s, :θ], [pad(:γ,k) for k = 1:$nr_γ])
        q = PosteriorFactorization(smin, s, θ, γ..., ids=ids)

    end
    
end
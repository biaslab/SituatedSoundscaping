mutable struct GSMM{T<:Float64}
    "number of Gaussians"
    n::Int
    "dimension of Gaussian"
    d::Int
    "weights (size n)"
    w::Array{T,1}
    "means (size d x n)"
    μ::Array{T,2}
    "covariances (size d x n)"
    Σ::Array{T,2}
    "covariances of messages to frequency domain (size d x n)"
    Σf::Array{T,2}
    function GSMM(w::AbstractArray{T,1}, μ::AbstractArray{T,2}, Λ::AbstractArray{T,2}) where T
        n = length(w)
        Σ = 1 ./ Λ
        abs(1 - sum(w)) < 1e-10 || error("weights do not sum to one")
        d = size(μ, 1)
        n == size(μ, 2) || error("Inconsistent number of means")
        (d,n) == size(Σ) || error("Inconsistent covar dimension")
        Σf = exp.(μ - Σ/2)

        new{T}(n, d, w, μ, Σ, Σf)
    end
end

Base.eltype(gsmm::GSMM{T}) where {T} = T



function train_GSMM(X::Array{Complex{Float64},2}, logX2::Array{Float64,2}, clusters::Int64; its::Int64=100)

    # stage 1 & 2: K-means & EM training
    gmm = GMM(clusters, logX2, nIter=50, nInit=500, kind=:diag)
    em!(gmm, logX2)

    # stage 3: variational bayesian training
    ξ, ϕ, f, qs, μ, ν, ps = initialize_parameters(gmm, logX2)
    for  _=1:its
        Estep!(X, ξ, ϕ, ν, μ, f, qs, ps)
        μ, ν, ps = Mstep(μ, ν, qs, ξ, ϕ)
    end

    # create GSMM object
    gsmm = GSMM(ps, μ, ν)

    # return GSMM
    return gsmm

end



# initialize parameters
function initialize_parameters(g::GMM, z::Array{Float64,2})
    
    # first t (time), then k (freq), then s (cluster)

    # initialize μ
    μ = collect(g.μ')
    
    # initialize ν
    ν = 1 ./ g.Σ'
    
    # initialize ps
    ps = g.w
    
    # initialize ξ
    ξ::Array{Float64,3} = repeat(z, outer=[1,1,g.n])

    # initialize ϕ
    ϕ = ones(g.nx, g.d, g.n)
    
    # initialize f
    f = Array{Float64, 3}(undef, g.nx, g.d, g.n)
    
    # initialize qs
    qs = ones(g.nx, g.n) / g.n
    
    return ξ, ϕ, f, qs, μ, ν, ps
    
end


function Estep!(X::Array{Complex{Float64},2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, ν::Array{Float64,2}, μ::Array{Float64,2}, f::Array{Float64,3}, qs::Array{Float64,2}, ps::Array{Float64,1})
    update_ξ!(ξ, ϕ, X, ν, μ)
    update_ϕ!(ϕ, ξ, X, ν)
    update_f!(f, ξ, ϕ, X, ν, μ)
    update_qs!(qs, ps, f)
end

function Mstep(μ::Array{Float64,2}, ν::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    update_μ!(μ, qs, ξ)
    update_ν!(ν, qs, μ, ξ, ϕ) 
    ps = update_ps(qs)
    return μ, ν, ps
end


function update_ξ!(ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})
    Threads.@threads for s = 1:size(ξ,3)
        for k = 1:size(ξ,2)
            for t = 1:size(ξ,1)
                @inbounds ξ[t,k,s] = ξ[t,k,s] + 1/ϕ[t,k,s]*( exp(-ξ[t,k,s])*abs2(X[t,k]) - ν[k,s]*ξ[t,k,s] + ν[k,s]*μ[k,s] - 1 ) 
            end
        end
    end
end

function update_ϕ!(ϕ::Array{Float64,3}, ξ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2})
    Threads.@threads for s = 1:size(ξ,3)
        for k = 1:size(ξ,2)
            for t = 1:size(ξ,1)
                @inbounds ϕ[t,k,s] = exp(-ξ[t,k,s])*abs2(X[t,k]) + ν[k,s]
            end
        end
    end
end

function update_f!(f::Array{Float64,3}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})
    Threads.@threads for s = 1:size(ξ,3)
        for k = 1:size(ξ,2)
            for t = 1:size(ξ,1)
                @inbounds f[t,k,s] = 0.5*log(ν[k,s]) - log(pi) - 0.5*log(ϕ[t,k,s]) - exp(-ξ[t,k,s] + 1/(2*ϕ[t,k,s])) * abs2(X[t,k]) - ξ[t,k,s] - ν[k,s]/2*(1/ϕ[t,k,s] + (ξ[t,k,s] - μ[k,s])^2) + 0.5
            end
        end
    end
end

function update_qs!(qs::Array{Float64,2}, ps::Array{Float64,1}, f::Array{Float64,3})
    for t = 1:size(f,1)
        for s = 1:size(f,3)
            @inbounds qs[t,s] = exp(sum(f[t,:,s]))*ps[s]
        end
        Zt = sum(qs[t,:])
        qs[t,:] = qs[t,:] / Zt
    end
end

function update_μ!(μ::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3})
    Threads.@threads for k = 1:size(ξ, 2)
        for s = 1:size(ξ, 3)
            @inbounds μ[k,s] = sum(qs[:,s].*ξ[:,k,s])/sum(qs[:,s])
        end
    end
end

function update_ν!(ν::Array{Float64,2}, qs::Array{Float64,2}, μ::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    Threads.@threads for k = 1:size(ξ, 2)
        for s = 1:size(ξ, 3)
            @inbounds ν[k,s] = sum(qs[:,s])/sum(qs[:,s] .* ((ξ[:,k,s] .- μ[k,s]).^2 + 1 ./ ϕ[:,k,s]))
        end
    end
end

function update_ps(qs::Array{Float64,2})::Array{Float64,1}
    return squeeze(sum(qs, dims=1)) / sum(qs)
end
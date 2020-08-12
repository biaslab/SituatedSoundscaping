# initialize parameters
function initialize_parameters(g, z)
    
    # initialize μ
    μ = collect(g.μ')
    
    # initialize ν
    ν = 1 ./ g.Σ'
    
    # initialize ps
    ps = g.w
    
    # initialize ξ
    ξ = repeat(transpose(z), outer=[1,1,g.n])

    # initialize ϕ
    ϕ = ones(g.d, g.nx, g.n)
    
    # initialize f
    f = Array{Float64, 3}(undef, g.d, g.nx, g.n)
    
    # initialize qs
    qs = ones(g.nx, g.n) / g.n
    
    return ξ, ϕ, f, qs, μ, ν, ps
    
end

function update_ξ(ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds ξ[k,t,s] = ξ[k,t,s] + 1/ϕ[k,t,s]*( exp(-ξ[k,t,s])*abs2(X[k,t]) - ν[k,s]*ξ[k,t,s] + ν[k,s]*μ[k,s] - 1 ) 
            end
        end
    end
    return ξ
end



function update_ϕ(ϕ::Array{Float64,3}, ξ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds ϕ[k,t,s] = exp(-ξ[k,t,s])*abs2(X[k,t]) + ν[k,s]
            end
        end
    end
    return ϕ
end

function update_f(f::Array{Float64,3}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds f[k,t,s] = 0.5*log(ν[k,s]) - log(pi) - 0.5*log(ϕ[k,t,s]) - exp(-ξ[k,t,s] + 1/(2*ϕ[k,t,s])) * abs2(X[k,t]) - ξ[k,t,s] - ν[k,s]/2*(1/ϕ[k,t,s] + (ξ[k,t,s] - μ[k,s])^2) + 0.5
            end
        end
    end
    return f
end



function update_qs(qs::Array{Float64,2}, ps::Array{Float64,1}, f::Array{Float64,3})
    F = 0.0
    for t = 1:size(f,2)
        for s = 1:size(f,3)
            @inbounds qs[t,s] = exp(sum(f[:,t,s]))*ps[s]
        end
        Zt = sum(qs[t,:])
        qs[t,:] = qs[t,:] / Zt
        F = F + log(Zt)
    end
    return qs, F
end



function update_μ(μ::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3})
    for k = 1:size(ξ, 1)
        for s = 1:size(ξ, 3)
            @inbounds μ[k,s] = sum(qs[:,s].*ξ[k,:,s])/sum(qs[:,s])
        end
    end
    return μ
end

function update_ν(ν::Array{Float64,2}, qs::Array{Float64,2}, μ::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    for k = 1:size(ξ, 1)
        for s = 1:size(ξ, 3)
            @inbounds ν[k,s] = sum(qs[:,s])/sum(qs[:,s] .* ((ξ[k,:,s] .- μ[k,s]).^2 + 1 ./ ϕ[k,:,s]))
        end
    end
    return ν
end

function update_ps(qs::Array{Float64,2})
    return squeeze(sum(qs, dims=1)) / sum(qs)
end

function Estep(X::Array{Complex{Float64},2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, ν::Array{Float64,2}, μ::Array{Float64,2}, f::Array{Float64,3}, qs::Array{Float64,2}, ps::Array{Float64,1})
    
    ξ = update_ξ(ξ, ϕ, X, ν, μ)
    ϕ = update_ϕ(ϕ, ξ, X, ν)
    f = update_f(f, ξ, ϕ, X, ν, μ)
    qs, F = update_qs(qs, ps, f)
    return ξ, ϕ, f, qs, F
end

function Mstep(μ::Array{Float64,2}, ν::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    μ = update_μ(μ, qs, ξ)
    ν = update_ν(ν, qs, μ, ξ, ϕ) 
    ps = update_ps(qs)
    return μ, ν, ps
end



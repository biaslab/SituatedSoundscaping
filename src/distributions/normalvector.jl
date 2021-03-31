import Statistics: mean, var, std, precision
import Base: *, length

export Dnormalvector, mean, var, std, precision, *, length

# distribution structure
struct Dnormalvector
    μ::Array{Float64,1}
    γ::Array{Float64,1}
    Dnormalvector(μ::Array{Float64,1}, γ::Array{Float64,1}) = new(μ, γ)
end

# statistics
mean(dist::Dnormalvector) = dist.μ
var(dist::Dnormalvector) = map(x -> 1/x, dist.γ)
std(dist::Dnormalvector) = 1 ./ sqrt.(dist.γ)
precision(dist::Dnormalvector) = dist.γ

# base functions
function *(dist1::Dnormalvector, dist2::Dnormalvector)
    γ = precision(dist1) + precision(dist2)
    μ = 1 ./γ .*(mean(dist1).*precision(dist1) + mean(dist2).*precision(dist2))
    return Dnormalvector(μ, γ)
end
length(dist::Dnormalvector) = length(dist.μ)
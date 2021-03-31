import Statistics: mean, var, std, precision
import Base: *, length

export Dcomplexnormalvector, mean, var, std, precision, *, length

# distribution structure
struct Dcomplexnormalvector
    μ::Array{Complex{Float64},1}
    γ::Array{Float64,1}
    Dcomplexnormalvector(μ::Array{Complex{Float64},1}, γ::Array{Float64,1}) = new(μ, γ)
end

# statistics
mean(dist::Dcomplexnormalvector) = dist.μ
var(dist::Dcomplexnormalvector) = 1 ./  dist.γ
std(dist::Dcomplexnormalvector) = 1 ./ sqrt.(dist.γ)
precision(dist::Dcomplexnormalvector) = dist.γ

#base functions
length(dist::Dcomplexnormalvector) = length(dist.dists)
function *(dist1::Dnormalvector, dist2::Dnormalvector)
    γ = precision(dist1) + precision(dist2)
    μ = 1/γ*(mean(dist1)*precision(dist1) + mean(dist2)*precision(dist2))
    return Dnormalvector(μ, γ)
end
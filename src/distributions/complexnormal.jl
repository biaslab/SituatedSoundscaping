import Statistics: mean, var, std, precision
import Base: *

export Dcomplexnormal, mean, std, var, precision, mode, entropy, H, *

# distribution structure
struct Dcomplexnormal
    μ::Complex{Float64}
    γ::Float64
    Dcomplexnormal(μ::Complex{Float64}, γ::Float64) = new(μ, γ)
    Dcomplexnormal(μ::Float64, γ::Float64) = Dcomplexnormal(μ+0.0im, γ)
    Dcomplexnormal(μ::Int64, γ::Float64) = Dcomplexnormal(float(μ), γ)
    Dcomplexnormal(μ::Float64, γ::Int64) = Dcomplexnormal(μ, float(γ))
    Dcomplexnormal(μ::Int64, γ::Int64) = Dcomplexnormal(float(μ), float(γ))
end

# statistics
mean(dist::Dcomplexnormal) = dist.μ
std(dist::Dcomplexnormal) = 1/sqrt(dist.γ)
var(dist::Dcomplexnormal) = 1/dist.γ
precision(dist::Dcomplexnormal) = dist.γ
mode(dist::Dcomplexnormal) = dist.μ

entropy(dist::Dcomplexnormal) = 1 + log(pi) - log(dist.γ)
H(dist::Dcomplexnormal) = entropy(dist)

# base functions
function *(dist1::Dcomplexnormal, dist2::Dcomplexnormal)
    γ = precision(dist1) + precision(dist2)
    μ = 1/γ*(mean(dist1)*precision(dist1) + mean(dist2)*precision(dist2))
    return Dcomplexnormal(μ, γ)
end
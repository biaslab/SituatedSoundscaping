import Statistics: mean, var, std, precision
import Base: *

export Dnormal, mean, std, var, precision, mode, entropy, H, *

# distribution structure
struct Dnormal
    μ::Float64
    γ::Float64
    Dnormal(μ::Float64, γ::Float64) = new(μ, γ)
    Dnormal(μ::Int64, γ::Float64) = new(float(μ), γ)
    Dnormal(μ::Float64, γ::Int64) = new(μ, float(γ))
    Dnormal(μ::Int64, γ::Int64) = new(float(μ), float(γ))
end

# statistics
mean(dist::Dnormal) = dist.μ
std(dist::Dnormal) = 1/sqrt(dist.γ)
var(dist::Dnormal) = 1/dist.γ
precision(dist::Dnormal) = dist.γ
mode(dist::Dnormal) = dist.μ

entropy(dist::Dnormal) = 0.5 + 0.5*log(2*pi) - 0.5*log(dist.γ)
H(dist::Dnormal) = entropy(dist)

# base functions
function *(dist1::Dnormal, dist2::Dnormal)
    γ = precision(dist1) + precision(dist2)
    μ = 1/γ*(mean(dist1)*precision(dist1) + mean(dist2)*precision(dist2))
    return Dnormal(μ, γ)
end
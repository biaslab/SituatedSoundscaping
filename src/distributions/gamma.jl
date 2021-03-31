using SpecialFunctions: gamma, digamma

import Statistics: mean, var, std, precision
import Base: *

export Dgamma, mean, std, var, precision, logmean, mode, entropy, H, *

# distribution structure
struct Dgamma
    a::Float64
    b::Float64
    Dgamma(a::Float64, b::Float64) = new(a, b)
    Dgamma(a::Int64, b::Float64) = new(float(a), b)
    Dgamma(a::Float64, b::Int64) = new(a, float(b))
    Dgamma(a::Int64, b::Int64) = new(float(a), float(b))
end

# statistics
mean(dist::Dgamma) = dist.a/dist.b
mode(dist::Dgamma) = dist.a >= 1 ? (dist.a-1)/dist.b : error("The parameter a of the gamma distribution is less than 1 and therefore the mode cannot be calculated.")
var(dist::Dgamma) = dist.a/dist.b^2
std(dist::Dgamma) = sqrt(dist.a)/dist.b
precision(dist::Dgamma) = dist.b^2/dist.a
logmean(dist::Dgamma) = digamma(dist.a) - digamma(dist.b)

entropy(dist::Dgamma) = dist.a - log(dist.b) + log(gamma(dist.a)) + (1-dist.a)*digamma(dist.a)
H(dist::Dgamma) = entropy(dist)

# Base functions
*(dist1::Dgamma, dist2::Dgamma) = Dgamma(dist1.a + dist2.a - 1, dist1.b + dist2.b)
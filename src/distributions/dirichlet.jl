using SpecialFunctions: digamma

import Statistics: mean
import Base: *, length

export Ddirichlet, mean, logmean, *, length

# distributions structure
struct Ddirichlet
    a::Array{Float64,1}
    Ddirichlet(a) = new(a)
end

# statistics
mean(dist::Ddirichlet) = dist.a ./ sum(dist.a)
logmean(dist::Ddirichlet) = digamma.(dist.a) .- digamma(sum(dist.a))
length(dist::Ddirichlet) = length(dist.a)

# base functions
*(dist1::Ddirichlet, dist2::Ddirichlet) = Ddirichlet(dist1.a + dist2.a .- 1)
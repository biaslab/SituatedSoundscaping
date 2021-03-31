using SpecialFunctions: digamma

import Statistics: mean
import Base: *

export Ddirichletmatrix, mean, logmean, *

# distribution structure
struct Ddirichletmatrix
    a::Array{Float64,2}
    Ddirichletmatrix(a::Array{Float64,2}) = new(a)
end

# statistics
mean(dist::Ddirichletmatrix) = dist.a./sum(dist.a,dims=1) # normalize columns
logmean(dist::Ddirichletmatrix) = digamma.(dist.a) .- digamma.(sum(dist.a,dims=1)) # Normalize columns

# base functions
*(dist1::Ddirichletmatrix, dist2::Ddirichletmatrix) = Ddirichletmatrix(dist1.a + dist2.a .- 1)
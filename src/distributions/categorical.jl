import Statistics: mean
import Base: *, length

export Dcategorical, mean, *, length

# distribution structure
struct Dcategorical
    p::Array{Float64,1}
    Dcategorical(p) = new(normalize_sum(p))
end

# statistics
mean(dist::Dcategorical) = dist.p

# base functions
function *(dist1::Dcategorical, dist2::Dcategorical)
    p = mean(dist1) .* mean(dist2)
    p_normalized = p ./ sum(p)
    return Dcategorical(p_normalized)
end
length(dist::Dcategorical) = length(dist.p)
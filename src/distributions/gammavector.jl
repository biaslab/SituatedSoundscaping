import Statistics: mean, var, std, precision
import Base: *, length

export Dgammavector, mean, std, var, precision, logmean, mode, *, length

# distribution structure
struct Dgammavector
    a::Array{Float64,1}
    b::Array{Float64,1}
    Dgammavector(a::Array{Float64,1}, b::Array{Float64,1}) = new(a, b)
end

# statistics
mean(dist::Dgammavector) = dist.a ./ dist.b
mode(dist::Dgammavector) = (dist.a .-1)./dist.b
var(dist::Dgammavector) = dist.a ./ dist.b.^2
std(dist::Dgammavector) = sqrt.(dist.a)./dist.b
precision(dist::Dgammavector) = dist.b.^2 ./dist.a
logmean(dist::Dgammavector) = digamma.(dist.a) - digamma.(dist.b)

# base functions
*(dist1::Dgammavector, dist2::Dgammavector) = Dgammavector(dist1.a + dist2.a .- 1, dist1.b + dist2.b)
length(dist::Dgammavector) = length(dist.a)

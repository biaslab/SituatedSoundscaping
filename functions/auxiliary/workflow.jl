# function that pads symbols with underscore and index (two digits)
#pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,2,'0')) # Left-pads a number with zeros, converts it to symbol and appends to symbol
pad(sym::Symbol, t::Int; len=2::Int) = sym*:_*Symbol(lpad(t,len,'0'))


# function that drops singleton dimensions
squeeze(a) = dropdims(a, dims = tuple(findall(size(a) .== 1)...))

# function that flattens arrays of arrays 
expand(a) = collect(Iterators.Flatten(a))

# function that first expands an array and then squeezes it
simplify(a) = squeeze(expand(a))

# fix compatibility issue with ForneyLab.jl and Distributions.jl
function compatibility_fix(algo)
    algo = replace(algo, "Univariate" => "ForneyLab.Univariate")
    algo = replace(algo, "Multivariate" => "ForneyLab.Multivariate")
    algo = replace(algo, "Gamma," => "ForneyLab.Gamma,")
    algo = replace(algo, "Bernoulli," => "ForneyLab.Bernoulli,")
    algo = replace(algo, "Wishart," => "ForneyLab.Wishart,")
    algo = replace(algo, "Beta," => "ForneyLab.Beta,")
    return algo
end

# function to collect an identity matrix
 Ic(dim::Int64) = 1*collect(I(dim))

# finds all step functions and runs them all once
function step_all!(data::Dict, marginals::Dict=Dict())
    # fetch different types of step functions
    functions = names(Main)[setdiff(findall(x -> occursin("step", String(x)), names(Main)), findall(x -> ("step!" == String(x)) || ("step_all!" == String(x)), names(Main)))]
    # invoke functions
    for func in functions
        Base.invokelatest(eval(func), data, marginals)
    end
    # return output
    return data, marginals
end

# create basis function of len with 1 at location loc
function em(len::Int, loc::Int)
    basis_vector = zeros(len)
    basis_vector[loc] = 1
    return basis_vector
end


function safeChol(A::Hermitian)
    # `safeChol(A)` is a 'safe' version of `chol(A)` in the sense
    # that it adds jitter to the diagonal of `A` and tries again if
    # `chol` raised a `PosDefException`.
    # Matrix `A` can be non-positive-definite in practice even though it
    # shouldn't be in theory due to finite floating point precision.
    # If adding jitter does not help, `PosDefException` will still be raised.
    L = similar(A)
    try
        L = cholesky(A)
        catch #Base.LinAlg.PosDefException
        # Add jitter to diagonal to break linear dependence among rows/columns.
        # The additive noise is input-dependent to make sure that we hit the
        # significant precision of the Float64 values with high probability.
        jitter = Diagonal(1e-13*(rand(size(A,1))) .* diag(A))
        L = cholesky(A + jitter)
    end
end
ForneyLab.unsafeMean(dist::ProbabilityDistribution{ForneyLab.Multivariate, GaussianWeightedMeanPrecision}) = inv(safeChol(Hermitian(dist.params[:w])))*dist.params[:xi]
unsafeMean(dist::ProbabilityDistribution{ForneyLab.Multivariate, GaussianWeightedMeanPrecision}) = inv(safeChol(Hermitian(dist.params[:w])))*dist.params[:xi]
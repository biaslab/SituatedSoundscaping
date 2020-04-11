# function that pads symbols with underscore and index (two digits)
pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,2,'0')) # Left-pads a number with zeros, converts it to symbol and appends to symbol

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
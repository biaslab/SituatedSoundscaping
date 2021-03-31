## data.jl Julia code to handle matrix-type data on disc
using Statistics
using Distributed
using FileIO

## A data handle, either in memory or on disk, perhaps even mmapped but I haven't seen any
## advantage of that.  It contains a list of either files (where the data is stored)
## or data units.  The point is, that in processing, these units can naturally be processed
## independently.

## The API is a dictionary of functions that help loading the data into memory
## Compulsory is: :load, useful is: :size
"""
`Data` is a type for holding an array of feature vectors (i.e., matrices), or references to
files on disk.  The data is automatically loaded when needed, e.g., by indexing.
"""
struct Data{T,VT<:Union{Matrix,AbstractString}}
    list::Vector{VT}
    function Data{T,VT}(list::Union{Vector{VT},Vector{Matrix{T}}}) where{T,VT}
        return new(list)
    end
end

Data(list::Vector{Matrix{T}}) where {T} = Data{T, eltype(list)}(list)
Data(list::Vector{S}, t::DataType) where {S<:AbstractString} = Data{t, S}(list)

DataOrMatrix{T} = Union{Data{T}, Matrix{T}}

## report the kind of Data structure from the type instance
kind(d::Data{T,S}) where {T,S<:AbstractString} = :file
kind(d::Data{T,Matrix{T}}) where {T} = :matrix
Base.eltype(d::Data{T}) where {T} = T

## constructor for a plain matrix.  rowvectors: data points x represented as rowvectors
Data(x::Matrix{T}) where {T} = Data(Matrix{T}[x])

## courtesy compatible save for a matrix
function FileIO.save(file::AbstractString, x::Matrix)
    save(file,"data", x)
end

function Base.getindex(x::Data, i::Int)
    if kind(x) == :matrix
        x.list[i]
    elseif kind(x) == :file
        myread(x.list[i])
    else
        error("Unknown kind")
    end
end

## A range as index (including [1:1]) returns a Data object, not the data.
function Base.getindex(x::Data{T,VT}, r::UnitRange{Int}) where {T,VT}
    Data{T, VT}(x.list[r])
end
function Base.lastindex(x::Data)
    length(x)
end

## define an iterator for Data
Base.length(x::Data) = length(x.list)
function Base.iterate(x::Data, state=1)
    count = state

    if(count > length(x.list))
        return nothing
    else
        return (x[count], count+1)
    end
end

## This function is like pmap(), but executes each element of Data on a predestined
## worker, so that file caching at the local machine is beneficial.
## This is _not_ dynamic scheduling, like pmap().
function dmap(f::Function, x::Data)
    if kind(x) == :file
        nₓ = length(x)
        w = workers()
        nw = length(w)
        worker(i) = w[1 .+ ((i-1) % nw)]
        results = Array{Any,1}(undef,nₓ)
        getnext(i) = x.list[i]
        load = x.API[:load]
        @sync begin
            for i = 1:nₓ
                @async begin
                    next = getnext(i)
                    results[i] = remotecall_fetch(s->f(load(s)), worker(i), next)
                end
            end
        end
        results
    elseif kind(x) == :matrix
        pmap(f, x)
    else
        error("Unknown kind")
    end
end

## this is like mapreduce(), but works on a data iterator
## for every worker, the data is reduced immediately, so that
## we only keep a list of (possibly large results) of the size
## of the array (and not the size of the data list)
function Base.mapreduce(f::Function, op::Function, x::Data)
    nₓ = length(x)
    nw = nworkers()
    results = Array{Any}(undef, nw) ## will contain pointers for parallel return value.
    valid = Any[false for i=1:nw] # can't use bitarray as parallel return value, must be pointers
    id=0
    nextid() = (id += 1)
    @sync begin
        for (wi,wid) in enumerate(workers())
            @async begin
                while true
                    i = nextid()
                    if i > nₓ
                        break
                    end
                    if kind(x) == :matrix
                        r = remotecall_fetch(f, wid, x[i])
                    else
                        r = remotecall_fetch(s->f(x.API[:load](s)), wid, x.list[i])
                    end
                    if valid[wi]
                        results[wi] = op(results[wi], r)
                    else
                        results[wi] = r
                        valid[wi] = true
                    end
                end
            end
        end
    end
    reduce(op, results[findall(valid)])
end

## stats: compute nth order stats for array (this belongs in stats.jl)
function stats(x::Matrix{T}, order::Int=2; kind=:diag, dim=1) where {T<:AbstractFloat}
    if dim==1
        n, d = size(x)
    else
        d, n = size(x)
    end
    if kind == :diag
        if order == 2
            return n, vec(sum(x, dims=dim)), vec(sum(abs2, x, dims=dim))
        elseif order == 1
            return n, vec(sum(x, dims=dim))
        else
            sx = zeros(T, order, d)
            for j=1:d
                for i=1:n
                    if dim==1
                        xi = xp = x[i,j]
                    else
                        xi = xp = x[j,i]
                    end
                    sx[1,j] += xp
                    for o=2:order
                        xp *= xi
                        sx[o,j] += xp
                    end
                end
            end
            return tuple([n, map(i->vec(sx[i,:]), 1:order)...]...)
        end
    elseif kind == :full
        order == 2 || error("Can only do covar starts for order=2")
        ## lazy implementation
        sx = vec(sum(x, dims=dim))
        sxx = x' * x
        return n, sx, sxx
    else
        error("Unknown kind")
    end
end

## Helper functions for stats tuples:
## This relies on sum(::Tuple), which sums over the elements of the tuple.
import Base: +
function +(a::Tuple, b::Tuple)
    length(a) == length(b) || error("Tuples must be of same length in addition")
    tuple(map(sum, zip(a,b))...)
end
Base.zero(t::Tuple) = map(zero, t)

## this function calls dmap as an option for parallelism
function stats(d::Data, order::Int=2; kind=:diag, dim=1)
    s = dmap(x->stats(x, order, kind=kind, dim=dim), d)
    if dim==1
        return reduce(+, s)
    else
        ## this is admittedly hairy: vertically concatenate each element of stats
        n = s[1][1]
        st = map(i->reduce(vcat, [x[i] for x in s]), 1+(1:order))
        return tuple(n, st...)
    end
end

## helper function to get summary statistics in traditional shape
function retranspose(x::Array, dim::Int)
    if dim==1
        return x'
    else
        return x
    end
end

## sum, mean, var
function Base.sum(d::Data)
    s = zero(eltype(d))
    for x in d
        s += sum(x)
    end
    return s
end

Base.sum(d::Data, dim::Int) = retranspose(stats(d,1, dim=dim)[2], dim)

function Statistics.mean(d::Data)
    n, sx = stats(d, 1)
    sum(sx) / (n*length(sx))
end

 function Statistics.mean(d::Data, dim::Int)
     n, sx = stats(d, 1, dim=dim)
     return retranspose(sx ./ n, dim)
end

function Statistics.var(d::Data)
    n, sx, sxx = stats(d, 2)
    n *= length(sx)
    ssx = sum(sx)                       # keep type stability...
    ssxx = sum(sxx)
    μ = ssx / n
    return (ssxx - n*μ^2) / (n - 1)
end

function Statistics.var(d::Data, dim::Int)
    n, sx, sxx = stats(d, 2, dim=dim)
    μ = sx ./ n
    return retranspose((sxx - n*μ.^2) ./ (n-1), dim)
end

function Statistics.cov(d::Data)
    n, sx, sxx = stats(d, 2, kind=:full)
    μ = sx ./ n
    (sxx - n*μ*μ') ./ (n-1)
end

Base.collect(d::Data) = hcat([x for x in d]...)

function Base.convert(::Type{Data{Td}}, d::Data{Ts}) where {Td,Ts}
    Td == Ts && return d
    if kind(d) == :file
        api = copy(d.API)
        _load = api[:load] # local copy
        api[:load] = x -> Array{Td}(_load(x))
        return Data(d.list, Td, api)
    elseif kind(d) == :matrix
        Data(Matrix{Td}[Matrix{Td}(x) for x in d.list])
    else
        error("Unknown kind")
    end
end


function Base.size(d::Data)
    d2 = 0
    for k in 1:length(d)
        d2 += mysize(d.list[k])[2]
    end
    d1 = mysize(d.list[1])[1]
    return (d1, d2)
end

function maxsize(d::Data)
    d2 = 0
    for k in 1:length(d)
        dtmp = mysize(d.list[k])[2]
        if dtmp > d2
            d2 = dtmp
        end
    end
    d1 = mysize(d.list[1])[1]
    return (d1, d2)
end

myread(f::String) = h5read(f, "data")::Array{Float64,2}
mysize(f::String) = h5read(f, "size")::Array{Int64,1}

function Base.size(d::Data, id::Int64)
    mysize(d.list[id])
end
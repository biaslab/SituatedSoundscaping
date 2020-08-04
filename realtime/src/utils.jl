# included functions:
#   squeeze(a::AbstractArray)
#   dB10toNum(a::Real)
#   numtodB10(a::Real)
#   dB20toNum(a::Real)
#   numtodB20(a::Real)



#   squeeze
#
#   info:
#       This function removes the singleton dimensions of an arbitrarily shaped array with arbitrary contents.
#
#   input arguments:
#       a::AbstractArray        - Array of arbitrary shape with arbitrary contents
#   output arguments:
#       _::AbstractArray         - Array of shape less or equal to the input array with same contents

function squeeze(A::AbstractArray{T,N}) where {T,N}

    # find singleton dimensions
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)

    A = dropdims(A; dims=singleton_dims)

    # return array with dropped dimensions
    return A

end



#   dB10toNum
#
#   info:
#       This function converts a value in dB (10 scale) to a number.
#
#   input arguments:
#       a::AbstractFloat        - Value in dB
#   output arguments:
#       _::AbstractFloat        - Normal value

function dB10toNum(a::Real)

    # perform conversion
    a = 10^(a/10)

    # return value
    return a::Float64

end


#   numtodB10
#
#   info:
#       This function converts a value to its valuein dB (10 scale).
#
#   input arguments:
#       a::AbstractFloat        - Normal value
#   output arguments:
#       _::AbstractFloat        - Value in dB

function numtodB10(a::Real)

    # perform conversion
    a = 10*log10(a)

    # return value
    return a::Float64

end


#   dB20toNum, dB10toNum!
#
#   info:
#       This function converts a value in dB (20 scale) to a number.
#
#   input arguments:
#       a::AbstractFloat        - Value in dB
#   output arguments:
#       _::AbstractFloat        - Normal value

function dB20toNum(a::Real)

    # perform conversion
    a = 10^(a/20)

    # return value
    return a::Float64

end

#   numtodB20
#
#   info:
#       This function converts a value to its valuein dB (20 scale).
#
#   input arguments:
#       a::AbstractFloat        - Normal value
#   output arguments:
#       _::AbstractFloat        - Value in dB

function numtodB20(a::Real)

    # perform conversion
    a = 20*log10(a)

    # return value
    return a::Float64

end

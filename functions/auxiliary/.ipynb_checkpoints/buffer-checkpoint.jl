mutable struct BufferData
    data_s::Array{Float64}
    data_t::Array{Float64}
    buffer_size::Int
    location::Int
    buffer_s::Array{Float64}
    buffer_t::Array{Float64}
    buffer_told::Array{Float64}
    
    function BufferData(data_s::Array{Float64}, data_t::Array{Float64}, buffer_size::Int)
        # check whether the dimensions of t and s match
        @assert size(data_s) == size(data_t)
        # create buffer object
        return new(data_s, data_t, buffer_size, 1, reverse(data_s[1:buffer_size]), reverse(data_t[1:buffer_size]), reverse(data_t[1:buffer_size]))
    end
    
end;

function step!(buffer::BufferData, step::Int)
    if buffer.location + step + buffer.buffer_size < length(buffer.data_t) 
        buffer.location += step
        buffer.buffer_s = reverse(buffer.data_s[buffer.location:(buffer.location + buffer.buffer_size - 1)])
        buffer.buffer_told = buffer.buffer_t
        buffer.buffer_t = reverse(buffer.data_t[buffer.location:(buffer.location + buffer.buffer_size - 1)])
        return true
    else
        return false
    end
end;

function len(buffer::BufferData, step::Int)
    return Int(floor( (length(buffer.data_s) - buffer.buffer_size) / step ))
end;
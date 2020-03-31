function calc_C(f::Array{Float64}, t::Array{Float64})
    
    # allocate space for c
    C = Array{Float64}(undef, length(t), 2*length(f)+1)
    
    # loop through time (rows) and fill matrix C
    for (idx, ti) in enumerate(t)
        C[idx, :] = cat(dims=1, [1.0], sin.(2*pi*f*ti), cos.(2*pi*f*ti))
    end
    
    # return matrix C
    return C
end;

function calc_C_block(f::Array{Float64}, t::Array{Float64}, BW::Array{Float64})
    
    # allocate space for c
    C = Array{Float64}(undef, length(t), 2*length(f)+1)
    
    # loop through time (rows) and fill matrix C
    for (idx, ti) in enumerate(t)
        tx = ti
        C[idx, :] = cat(dims=1, [1.0], sin.(2*pi*f*ti).*sinc.(BW*ti)./BW, cos.(2*pi*f*ti).*sinc.(BW*ti)./BW)
    end
    
    # return matrix C
    return C
end;
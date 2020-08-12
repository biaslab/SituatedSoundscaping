function update_X(x::Float64, X::Array{Float64,1})
    # define order
    order = length(X)
    
    # define matrix C
    C = [zeros(1,order)
         Ic(order-1) zeros(order-1, 1)]
    
    # define matrix D
    D = [1
         zeros(order-1, 1)]
    
    # calculate new X
    X = C*X + D*x
    
    # return X
    return squeeze(X)
end

function update_Y(X::Array{Float64,1}, Y::Array{Float64,1}, z::Complex{Float64})
    # define matrix A
    A = [2*real(z) -abs(z)^2 0
         1 0         0
         0 1         0         ]
    
    # define matrix B
    B = [abs(z)^2 -2*real(z) 1
         0         0          0
         0         0          0]
    
    # calculate new Y
    Ynew = A*Y + B*X
    
    # return Ynew
    return squeeze(Ynew)
end

function allpass(x::Float64, Y::Vector{Array{Float64,1}}, z::Complex{Float64})
    
    # update zero'th Y by simply buffering
    Y[1] = update_X(x, Y[1])
    
    # loop through all all-pass filters and update all other Y outputs
    for k = 2:length(Y)
        
        # update corresponding Y
        Y[k] = update_Y(Y[k-1], Y[k], z)
        
    end
    
    # return memory vector for outputs
    return Y
    
end
function allpass(x::Float64, Y::Vector{Array{Float64,1}}, z::Float64) 
    allpass(x, Y, z+0*im)
end

function allpass_update_matrix(order::Int, z::Complex{Float64})
    # this function calculates the large update matrix for an all-pass filter
        
    # specify matrices
    A = [2*real(z) -abs(z)^2 0
         1 0         0
         0 1         0         ]
    B = [abs(z)^2 -2*real(z) 1
         0         0          0
         0         0          0]
    C = [0 0 0
         1 0 0
         0 1 0]
    D = zeros(3,1)
    D[1] = 1
    
    # create matrix T
    T = [((k<=l) ? B^(l-k) : 0)*((k==1) ? C : A) for l=1:order, k=1:order]
    T = hvcat(order, permutedims(T,[2,1])...)
    
    # create matrix u
    #u = repeat(D, order)
    u = [B^(k-1)*D for k=1:order]
    u = vcat(u...)
    
    return T, u
end
function allpass_update_matrix(order::Int, z::Float64) 
    allpass_update_matrix(order, z+0*im)
end

function allpass_update_matrix2(order::Int, z::Float64)
    # this function calculates the large update matrix for an all-pass filter
        
    # specify matrices
    A = [z 0
         1 0]
    B = [-z 1
         0  0]
    C = [0 0
         1 0]
    D = zeros(2,1)
    D[1] = 1
    
    # create matrix T
    T = [((k<=l) ? B^(l-k) : 0)*((k==1) ? C : A) for l=1:order, k=1:order]
    T = hvcat(order, permutedims(T,[2,1])...)
    
    # create matrix u
    #u = repeat(D, order)
    u = [B^(k-1)*D for k=1:order]
    u = vcat(u...)
    
    return T, u
end

function allpass_update(x::Float64, Y::Array{Float64,2}, order::Int, z::Complex{Float64}; T=nothing, u=nothing)
    
    # define update matrices if not defined
    if (T == nothing) | (u == nothing)
        T, u = allpass_update_matrix(order, z)
    end
    
    # update 
    Ynew = T*Y + u*x
    
    # select taps
    taps = Ynew[1:3:length(Ynew)]
    
    # return new Y
    return Ynew, taps
end
function allpass_update(x::Float64, Y::Array{Float64,2}, order::Int, z::Float64; T=nothing, u=nothing) 
    allpass_update(x, Y, order, z+0*im, T=T, u=u)
end



function allpass_update_matrix_segment(order::Int, input_length::Int, z::Complex{Float64})
    # this function calculates the large update matrix for an all-pass filter
        
    # specify matrices
    A = [2*real(z) -abs(z)^2 0
         1 0         0
         0 1         0         ]
    B = [abs(z)^2 -2*real(z) 1
         0         0          0
         0         0          0]
    C = [0 0 0
         1 0 0
         0 1 0]
    D = zeros(3,1)
    D[1] = 1
    
    # create matrix T
    T = [((k<=l) ? B^(l-k) : 0)*((k==1) ? C : A) for l=1:order, k=1:order]
    T = hvcat(order, permutedims(T,[2,1])...)
    
    # create matrix u
    #u = repeat(D, order)
    u = [B^(k-1)*D for k=1:order]
    u = vcat(u...)
    
    # convert for use with multiple inputs
    W = T^input_length
    U = [W^(k-1)*u for k=1:input_length]
    U = hcat(U...)
    
    # create selection matrix S
    S = zeros(order, 3*order)
    for k = 1:order
        S[k, 1+3*(k-1)] = 1
    end
    
    return W, U, S
end
function allpass_update_matrix_segment(order::Int, input_length::Int, z::Float64) 
    # this function calculates the large update matrix for an all-pass filter
        
    # specify matrices
    A = [z 0
         1 0]
    B = [-z 1
         0  0]
    C = [0 0
         1 0]
    D = zeros(2,1)
    D[1] = 1
    
    # create matrix T
    T = [((k<=l) ? B^(l-k) : 0)*((k==1) ? C : A) for l=1:order, k=1:order]
    T = hvcat(order, permutedims(T,[2,1])...)
    
    # create matrix u
    #u = repeat(D, order)
    u = [B^(k-1)*D for k=1:order]
    u = vcat(u...)
    
    # convert for use with multiple inputs
    W = T^input_length
    U = [W^(k-1)*u for k=1:input_length]
    U = hcat(U...)
    
    # create selection matrix S
    S = zeros(order, 2*order)
    for k = 1:order
        S[k, 1+2*(k-1)] = 1
    end
    
    return W, U, S
end

function warp_fft(signal::Array{Float64, 1}, z::Float64, len::Int64; step_size=1)
    # get update matrices
    T, u = allpass_update_matrix2(len, z)
    
    # create current hidden states of filter
    Y = zeros(2*len)
    
    # create array for outputs
    y = Array{Complex{Float64}, 2}(undef, Int(floor((length(signal) - len)/step_size)), len)
    
    # update filter
    for k = 1:Int(floor((length(signal) - len - 1)/step_size))
        
        # update filter
        for i = 1:step_size
            Y = T*Y + u*signal[(k-1)*step_size + i]
        end
        
        # calculate FFT and save
        y[k,:] = FFTW.fft(Y[1:2:end])
        
    end
    
    return y
    
    
end
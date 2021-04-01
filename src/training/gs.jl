export train_gs

function update_z!(q_μ::Array{Dnormalvector,1}, q_γ::Array{Dgammavector,1}, q_ξ::Dnormalvector, U::Array{Float64,1}, tmp::Array{Float64,1})
    U .= 0
    for k in 1:length(q_μ)
        @. tmp = q_μ[k].μ
        @. tmp -= q_ξ.μ
        tmp .^= 2
        @. tmp += q_ξ.γ
        @. tmp += q_μ[k].γ
        @. tmp *= -0.5*q_γ[k].a/q_γ[k].b
        @. tmp += 0.5*(digamma(q_γ[k].a) - digamma(q_γ[k].b))
        tmp .-= 0.5*log(2*pi)

        U[k] = sum(tmp)
        #U[k] = -sum(0.5*log(2*pi) .- 0.5*logmean(q_γ[k]) .+ 0.5*mean(q_γ[k]).*(var(q_μ[k]) + var(q_ξ) + (mean(q_μ[k]) - mean(q_ξ)).^2))
    end
    softmax!(U)
end

# update alpha
function update_qa!(data::Data, q_a::Ddirichlet, v_a_down::Ddirichlet, q_μ::Array{Dnormalvector,1}, q_γ::Array{Dgammavector,1}, nr_freqs::Int64, it::Int64)
    v_z_down = Dcategorical(exp.(logmean(q_a)) ./ sum(exp.(logmean(q_a))))
    q_a_p = v_a_down.a    # clear memory and only keep prior
    m_tmp = zeros(nr_freqs)
    freq1 = ones(nr_freqs)
    U = zeros(length(q_μ))
    tmp = zeros(nr_freqs)

    # smartly allocate memory for data items
    d = zeros(Complex{Float64}, maxsize(data)) 
    dsize = [1, 1]

    # loop through data files
    @showprogress "iteration "*string(it)*"a" for n in 1:length(data)

        # load data
        f = h5open(data.list[n], "r")
        dsize .= size(f["data_complex"])::Tuple{Int64, Int64}
        d[1:dsize[1], 1:dsize[2]] .= HDF5.readmmap(f["data_complex"], Complex{Float64});

        # loop through data samples
        @inbounds for k in 1:dsize[2]

            # upward message from gaussianscale node toward gaussian
            @inbounds for ki in 1:nr_freqs
                m_tmp[ki] = log(1e-10 + abs2(d[ki,k]))
            end
            q_ξ = Dnormalvector(m_tmp, freq1) # approx

            # update q_z
            update_z!(q_μ, q_γ, q_ξ, U, tmp)
            # q_z = v_z_up * v_z_down

            # calculate message towards dirichlet and update new cumulative variable
            U .*= v_z_down.p
            normalize_sum!(U)
            q_a_p .+= U

        end

        # close data file
        close(f)

    end
end

function update_μγ(data::Data, v_z_down::Dcategorical, v_μ_down::Array{Dnormalvector}, v_γ_down::Array{Dgammavector}, nr_freqs::Int64, it::Int64)
    q_μ_new = deepcopy(v_μ_down)
    q_μ = deepcopy(v_μ_down)
    q_γ_new = deepcopy(v_γ_down)
    q_γ = deepcopy(v_γ_down)
    U = zeros(length(q_μ_new))
    tmp = zeros(nr_freqs)
    freq1 = ones(nr_freqs)
    nr_mixtures = length(q_μ)

    m_μ_tmp = zeros(nr_freqs)
    g_μ_tmp = zeros(nr_freqs)
    a_γ_tmp = zeros(nr_freqs)
    b_γ_tmp = zeros(nr_freqs)

    # smartly allocate memory for data items
    d = zeros(Complex{Float64}, maxsize(data)) 
    dsize = [1, 1]

    @showprogress "iteration "*string(it)*"b - " for n in 1:length(data)

        # load data
        f = h5open(data.list[n], "r")
        dsize .= size(f["data_complex"])::Tuple{Int64, Int64}
        d[1:dsize[1], 1:dsize[2]] .= HDF5.readmmap(f["data_complex"], Complex{Float64});

        for k in 1:dsize[2]

            # upward message from gaussianscale node toward gaussian
            v_ξ_up = Dnormalvector(log.(1e-10 .+ abs2.(d[:,k])), freq1)
            q_ξ = v_ξ_up

            # update q_z
            update_z!(q_μ, q_γ, q_ξ, U, tmp)
            U .*= v_z_down.p
            normalize_sum!(U)

            # calculate message towards means and precisions
            for ki in 1:nr_mixtures
                q_μ_new[ki] *= Dnormalvector(mean(q_ξ), U[ki]*mean(q_γ[ki]))
                q_γ_new[ki] *= Dgammavector(1.0 .+ U[ki]*0.5*freq1, U[ki]*0.5*(var(q_μ[ki]) + var(q_ξ) + (mean(q_μ[ki]) - mean(q_ξ)).^2))
            end

        end

    end

    return q_μ_new, q_γ_new

end

function GaussianScaleVMP(data::Data, means::Array{Float64,2}, covs::Array{Float64,2}, πk::Array{Float64,1}; nr_iterations=10::Int64)
    (nr_freqs, nr_mixtures) = size(means)

    precs = 1 ./ covs

    # initialization
    v_μ_down = Array{Dnormalvector, 1}(undef, nr_mixtures)
    v_γ_down = Array{Dgammavector, 1}(undef, nr_mixtures)
    for m in 1:nr_mixtures
        v_μ_down[m] = Dnormalvector(means[:,m], precs[:,m])
        v_γ_down[m] = Dgammavector(precs[:,m].^2 ./ 1e-3, precs[:,m] ./ 1e-3)
    end
    q_μ = deepcopy(v_μ_down)
    q_γ = deepcopy(v_γ_down)
    q_a = Ddirichlet(πk .* size(data)[2])
    v_a_down = deepcopy(q_a)
    v_z_down = Dcategorical(exp.(logmean(q_a)) ./ sum(exp.(logmean(q_a))))

    for it in 1:nr_iterations
        update_qa!(data, q_a, v_a_down, q_μ, q_γ, nr_freqs, it)
        v_z_down = Dcategorical(exp.(logmean(q_a)) ./ sum(exp.(logmean(q_a))))
        q_μ, q_γ = update_μγ(data, v_z_down, v_μ_down, v_γ_down, nr_freqs, it)
    end

    return q_μ, q_γ, q_a

end



function train_gs(model_name::String, data::Data, means::Array{Float64,2}, covs::Array{Float64,2}, πk::Array{Float64,1}; nr_iterations=10::Int64)

    # fetch dimensions 
    (nr_frequencies, nr_mixtures) = size(means)
    nr_files = length(data)

    # filename
    filename = model_name*"_"*string(nr_frequencies)*"_"*string(nr_mixtures)*"_"*string(nr_files)*".h5"

    # check if model exists
    if isfile(filename)

        @info "GS model already trained."

        # load model 
        f = h5open(filename, "r")
        mean_of_mean = HDF5.read(f["mean_of_mean"], Float64);
        prec_of_mean = HDF5.read(f["prec_of_mean"], Float64);
        a_of_prec = HDF5.read(f["a_of_prec"], Float64);
        b_of_prec = HDF5.read(f["b_of_prec"], Float64);
        a = HDF5.read(f["alpha"], Float64);
        close(f)

        # create distributions
        q_μ = [Dnormalvector(mean_of_mean[:,k], prec_of_mean[:,k]) for k=1:nr_mixtures]
        q_γ = [Dgammavector(a_of_prec[:,k], b_of_prec[:,k]) for k=1:nr_mixtures]
        q_a = Ddirichlet(a)

    else

        # initialize clusters 
        q_μ, q_γ, q_a =  GaussianScaleVMP(data, means, covs, πk; nr_iterations=nr_iterations)

        # save model
        f = h5open(filename, "w")
        HDF5.write(f, "mean_of_mean", hcat(mean.(q_μ)...));
        HDF5.write(f, "prec_of_mean", hcat(precision.(q_μ)...));
        HDF5.write(f, "a_of_prec", hcat([q_γ[k].a for k in 1:length(q_γ)]...));
        HDF5.write(f, "b_of_prec", hcat([q_γ[k].b for k in 1:length(q_γ)]...));
        HDF5.write(f, "alpha", q_a.a);
        close(f)
        
    end 

    return q_μ, q_γ, q_a

end
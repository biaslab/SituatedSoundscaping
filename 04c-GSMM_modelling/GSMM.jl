function trainGMM(clusters::Int64, y::Array{Float64,2}, kind::Symbol)
    if (kind == :full) & isposdef(y)
        g = GMM(clusters, y, kind=kind)
        p = GMMprior(g.d, 0.1, 1.0)  ## set α0=0.1 and β0=1, and other values to a default
        v = VGMM(g, p) ## initialize variational GMM v with g
        em!(v, y)
    elseif (kind == :full) & ~isposdef(y)
        println("data matrix not posdef, going for diagonal GMM")
        g = GMM(clusters, y, kind=:diag)
        em!(g, y)
    else
        g = GMM(clusters, y, nIter=50, nInit=100, kind=:diag)
        em!(g, y)
    end
    return g
end;

# initialize parameters
function initialize_parameters(g, z)
    
    # initialize μ
    μ = collect(g.μ')
    
    # initialize ν
    ν = 1 ./ g.Σ'
    
    # initialize ps
    ps = g.w
    
    # initialize ξ
    ξ = repeat(transpose(z), outer=[1,1,g.n])

    # initialize ϕ
    ϕ = ones(g.d, g.nx, g.n)
    
    # initialize f
    f = Array{Float64, 3}(undef, g.d, g.nx, g.n)
    
    # initialize qs
    qs = ones(g.nx, g.n) / g.n
    
    return ξ, ϕ, f, qs, μ, ν, ps
    
end

function update_ξ(ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds ξ[k,t,s] = ξ[k,t,s] + 1/ϕ[k,t,s]*( exp(-ξ[k,t,s])*abs2(X[k,t]) - ν[k,s]*ξ[k,t,s] + ν[k,s]*μ[k,s] - 1 ) 
            end
        end
    end
    return ξ
end

function update_ϕ(ϕ::Array{Float64,3}, ξ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds ϕ[k,t,s] = exp(-ξ[k,t,s])*abs2(X[k,t]) + ν[k,s]
            end
        end
    end
    return ϕ
end

function update_f(f::Array{Float64,3}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, X::Array{Complex{Float64},2}, ν::Array{Float64,2}, μ::Array{Float64,2})::Array{Float64,3}
    for k = 1:size(ξ,1)
        for t = 1:size(ξ,2)
            for s = 1:size(ξ,3)
                @inbounds f[k,t,s] = 0.5*log(ν[k,s]) - log(pi) - 0.5*log(ϕ[k,t,s]) - exp(-ξ[k,t,s] + 1/(2*ϕ[k,t,s])) * abs2(X[k,t]) - ξ[k,t,s] - ν[k,s]/2*(1/ϕ[k,t,s] + (ξ[k,t,s] - μ[k,s])^2) + 0.5
            end
        end
    end
    return f
end

function update_qs(qs::Array{Float64,2}, ps::Array{Float64,1}, f::Array{Float64,3})
    F = 0.0
    for t = 1:size(f,2)
        for s = 1:size(f,3)
            @inbounds qs[t,s] = exp(sum(f[:,t,s]))*ps[s]
        end
        Zt = sum(qs[t,:])
        qs[t,:] = qs[t,:] / Zt
        F = F + log(Zt)
    end
    return qs, F
end

function update_μ(μ::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3})
    for k = 1:size(ξ, 1)
        for s = 1:size(ξ, 3)
            @inbounds μ[k,s] = sum(qs[:,s].*ξ[k,:,s])/sum(qs[:,s])
        end
    end
    return μ
end

function update_ν(ν::Array{Float64,2}, qs::Array{Float64,2}, μ::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    for k = 1:size(ξ, 1)
        for s = 1:size(ξ, 3)
            @inbounds ν[k,s] = sum(qs[:,s])/sum(qs[:,s] .* ((ξ[k,:,s] .- μ[k,s]).^2 + 1 ./ ϕ[k,:,s]))
        end
    end
    return ν
end

function update_ps(qs::Array{Float64,2})
    return squeeze(sum(qs, dims=1)) / sum(qs)
end

function Estep(X::Array{Complex{Float64},2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3}, ν::Array{Float64,2}, μ::Array{Float64,2}, f::Array{Float64,3}, qs::Array{Float64,2}, ps::Array{Float64,1})
    
    ξ = update_ξ(ξ, ϕ, X, ν, μ)
    ϕ = update_ϕ(ϕ, ξ, X, ν)
    f = update_f(f, ξ, ϕ, X, ν, μ)
    qs, F = update_qs(qs, ps, f)
    return ξ, ϕ, f, qs, F
end

function Mstep(μ::Array{Float64,2}, ν::Array{Float64,2}, qs::Array{Float64,2}, ξ::Array{Float64,3}, ϕ::Array{Float64,3})
    μ = update_μ(μ, qs, ξ)
    ν = update_ν(ν, qs, μ, ξ, ϕ) 
    ps = update_ps(qs)
    return μ, ν, ps
end

function trainGSMM_paper(clusters::Int64, y::Array{Float64,2}, fr::Array{Complex{Float64},2})
    g = trainGMM(clusters, y, :diag)
    ξ, ϕ, f, qs, μ, ν, ps = initialize_parameters(g, y);
    
    X = collect(transpose(fr))
    Fmem = []
    for _= 1:100
        ξ, ϕ, f, qs, F = Estep(X, ξ, ϕ, ν, μ, f, qs, ps)
        μ, ν, ps = Mstep(μ, ν, qs, ξ, ϕ)
        push!(Fmem, F)
    end
    return μ, ν, ps, Fmem
end


# prevent posdef errors (Marco his comment on ForneyLab issue #86)
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
ForneyLab.unsafeMean(dist::ProbabilityDistribution{Multivariate, GaussianWeightedMeanPrecision}) = inv(safeChol(Hermitian(dist.params[:w])))*dist.params[:xi]
unsafeMean(dist::ProbabilityDistribution{Multivariate, GaussianWeightedMeanPrecision}) = inv(safeChol(Hermitian(dist.params[:w])))*dist.params[:xi]

function build_fg_offline(nr_freqs::Int64, nr_clusters::Int64, nr_samples::Int64, bufsize::Int64, Σ_meas::Array{Float64,2})
    
    # create factor graph
    fg = FactorGraph()

    # create distionary for variables
    vars = Dict{Symbol, Variable}()

    # specify distribution over the selection variables
    @RV vars[:π] ~ ForneyLab.Dirichlet(placeholder(:α_π, dims=(nr_clusters,)))

    # create mixture components
    for k = 1:nr_clusters

        # specify distribution over precision matrix
        @RV vars[pad(:w,k)] ~ Wishart(placeholder(pad(:V_w,k), dims=(nr_freqs,nr_freqs)), placeholder(pad(:nu_w,k)))

        # specify distribution over mean
        @RV vars[pad(:m,k)] ~ GaussianMeanPrecision(placeholder(pad(:μ_m,k), dims=(nr_freqs,)), vars[pad(:w,k)])

    end

    # create sample-dependent random variables
    for k = 1:nr_samples

        # specify distribution over selection variable
        @RV vars[pad(:z,k)] ~ Categorical(vars[:π])

        # create gaussian mixture model
        @RV vars[pad(:ξ,k)] ~ GaussianMixture(vars[pad(:z,k)], expand([[vars[pad(:m,ki)], vars[pad(:w,ki)]] for ki=1:nr_clusters])...)

        # log-power to complex fourier coefficients transform
        @RV vars[pad(:Xc,k)] ~ ComplexHGF(vars[pad(:ξ,k)])

        # complex fourier coefficients to real and imaginary parts concatenated
        @RV vars[pad(:Xr,k)] ~ ComplexToReal(vars[pad(:Xc,k)])

        # probabilistic Fourier transform
        @RV vars[pad(:x,k)] = placeholder(pad(:C,k), dims=(bufsize, 2*(nr_freqs)))*vars[pad(:Xr,k)]

        # observation model 
        @RV vars[pad(:y,k)] ~ GaussianMeanVariance(vars[pad(:x,k)], Σ_meas)

        # observation
        placeholder(vars[pad(:y,k)], :y, index=k, dims=(bufsize,))
    end
    
    return vars
    
end
    
function recognitionfactorization_offline(vars, nr_clusters, nr_samples)
    
    # specify ids for the posterior factorization
    q_ids = vcat(:Π,
                 expand([[pad(:M,k), pad(:W,k)] for k=1:nr_clusters]),
                 :Z, :Xc, :Ξ)

    # specify posterior factorization
    q = PosteriorFactorization(vars[:π], 
                               expand([[vars[pad(:m,k)], vars[pad(:w,k)]] for k=1:nr_clusters])...,
                               [vars[pad(:z,k)] for k=1:nr_samples],
                               [vars[pad(:Xc,k)] for k=1:nr_samples],
                               [vars[pad(:ξ,k)] for k=1:nr_samples],
                               ids=q_ids)
    
    # return factorization
    return q
end

function generate_algorithm(q)
    # generate the inference algorithm
    algo = variationalAlgorithm(q)

    # Generate source code
    source_code = algorithmSourceCode(algo)

    # Load algorithm
    eval(Meta.parse(source_code))
    
    return source_code
    
end

function prepare_data(stepsize, bufsize, y, t; shuffle=true)
    
    nr_samples = Int((length(y)-bufsize)/stepsize +1)
    
    shuff = randperm(nr_samples)
    
    y_samples = [y[(k-1)*stepsize+1:(k-1)*stepsize+bufsize] for k = 1:nr_samples];
    t_samples = [t[(k-1)*stepsize+1:(k-1)*stepsize+bufsize] for k = 1:nr_samples];
    if shuffle
        y_samples = y_samples[shuff]
        t_samples = t_samples[shuff]
    end
    
    return y_samples, t_samples
end

function create_data_dict_offline(y, t, g, fi, bufsize)
        
    nr_clusters = g.n
    nr_samples = length(t)
    nr_freqs = g.d
    
    # create data dictionary
    data = Dict{Symbol, Union{Int64, Array{Float64,1}, Array{Float64,2}, Array{Array{Float64,1},1}}}()

    # specify input data and measurement noise
    data[:y] = y

    # specify priors over class probabilities
    data[:α_π] = 1.0*ones(nr_clusters)

    # specify priors over clusters
    for k = 1:nr_clusters
        data[pad(:μ_m,k)] = g.μ[k,:]
        data[pad(:nu_w,k)] = (nr_freqs)
        data[pad(:V_w,k)] = diagm(1 ./g.Σ[k,:]) / (nr_freqs)
    end

    # specify probabilistic fourier matrices
    for k = 1:nr_samples
        data[pad(:C,k)] = 1/bufsize*hcat(cos.(2*pi*fi*t[k]')', sin.(2*pi*fi*t[k]')')
    end
    
    return data
end

function create_marginals_dict_offline(z, g, data)
    
    nr_clusters = g.n
    nr_samples = size(z,1)
    nr_freqs = g.d
    
    # create marginals dictionary
    marginals = Dict{Symbol, ProbabilityDistribution}()

    # specify marginas over class probabilities
    marginals[:vars_π] = vague(ForneyLab.Dirichlet, nr_clusters)

    # specify marginals over clusters
    for k = 1:nr_clusters
        marginals[pad(:vars_m,k)] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=data[pad(:μ_m,k)]::Array{Float64,1}, w=data[pad(:V_w,k)]::Array{Float64,2}*data[pad(:nu_w,k)]::Int64)
        marginals[pad(:vars_w,k)] = ProbabilityDistribution(MatrixVariate, ForneyLab.Wishart, v=data[pad(:V_w,k)]::Array{Float64,2}, nu=data[pad(:nu_w,k)]::Int64)
    end

    # specify marginals over samples
    for k = 1:nr_samples
        marginals[pad(:vars_z,k)] = ProbabilityDistribution(Categorical, p=weights(g))
        marginals[pad(:vars_Xc,k)] = ProbabilityDistribution(Multivariate, ComplexNormal, μ=zeros(nr_freqs) .+ 0.0im, Γ=1e10*Ic(nr_freqs).+0.0im, C=mat(0.0+0.0im))
        marginals[pad(:vars_ξ,k)] = ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=z[k,:], v=1*Ic(nr_freqs))
    end
    
    return marginals
    
end

function perform_inference_offline(data::Dict{Symbol, Union{Int64, Array{Float64,1}, Array{Float64,2}, Array{Array{Float64,1},1}}}, marginals::Dict{Symbol, ProbabilityDistribution}, nr_clusters::Int64; nr_its=10::Int64)

    # create progress bar
    p = Progress(nr_its)

    # perform iterations
    for i = 1:nr_its

        # perform updates
        Base.invokelatest(stepXc!, data, marginals)
        Base.invokelatest(stepΞ!, data, marginals)
        Base.invokelatest(stepZ!, data, marginals)
        Base.invokelatest(stepΠ!, data, marginals) 
        for k = 1:nr_clusters
            Base.invokelatest(getfield(Main, Symbol("stepM_"*string(k,pad=2)*"!")), data, marginals)
            Base.invokelatest(getfield(Main, Symbol("stepW_"*string(k,pad=2)*"!")), data, marginals)
        end

        # update progress bar
        next!(p)

    end
    return data, marginals
end

function trainGSMM_FL_offline(clusters::Int64, z::Array{Float64,2}, y::Array{Float64,1}, t::Array{Float64,1}, stepsize::Int64, bufsize::Int64, Σ_meas::Array{Float64,2}, fi::Array{Float64,1}; shuffle=true)
    
    # train GMM on deterministic log-power spectrum
    g = GMM(nr_clusters, z, nIter=50, nInit=100, kind=:diag)
    em!(g, z)

    nr_freqs = size(z,2)
    nr_samples = size(z,1)
    
    vars = build_fg_offline(nr_freqs, nr_clusters, nr_samples, bufsize, Σ_meas)
    
    q = recognitionfactorization_offline(vars, clusters, nr_samples)
    
    generate_algorithm(q)
    
    ys, ts = prepare_data(stepsize, bufsize, y, t; shuffle=shuffle)
    
    data = create_data_dict_offline(ys, ts, g, fi, bufsize)

    marginals = create_marginals_dict_offline(z, g, data)
    
    return data, marginals, g
end



function build_fg_online(nr_freqs::Int64, nr_clusters::Int64, nr_samples::Int64, bufsize::Int64, Σ_meas::Array{Float64,2})
    
    # create factor graph
    fg = FactorGraph()

    # create distionary for variables
    vars = Dict{Symbol, Variable}()

    # create mixture components
    for k = 1:nr_clusters

        # specify distribution over precision matrix
        @RV vars[pad(:w,k)] ~ Wishart(placeholder(pad(:V_w,k), dims=(nr_freqs,nr_freqs)), placeholder(pad(:nu_w,k)))

        # specify distribution over mean
        @RV vars[pad(:m,k)] ~ GaussianMeanPrecision(placeholder(pad(:μ_m,k), dims=(nr_freqs,)), vars[pad(:w,k)])

    end

    @RV vars[:π] ~ Dirichlet(placeholder(:α_π, dims=(nr_clusters,)))

    # specify distribution over selection variable
    @RV vars[:z] ~ Categorical(vars[:π])

    # create gaussian mixture model
    @RV vars[:ξ] ~ GaussianMixture(vars[:z], expand([[vars[pad(:m,ki)], vars[pad(:w,ki)]] for ki=1:nr_clusters])...)

    # log-power to complex fourier coefficients transform
    @RV vars[:Xc] ~ ComplexHGF(vars[:ξ])

    # complex fourier coefficients to real and imaginary parts concatenated
    @RV vars[:Xr] ~ ComplexToReal(vars[:Xc])

    # probabilistic Fourier transform
    @RV vars[:x] = placeholder(:C, dims=(bufsize, 2*(nr_freqs)))*vars[:Xr]

    # observation model 
    @RV vars[:y] ~ GaussianMeanVariance(vars[:x], Σ_meas)

    # observation
    placeholder(vars[:y], :y, dims=(bufsize,))
   
    return vars
end


    
function recognitionfactorization_online(vars, nr_clusters, nr_samples)

    # specify ids for the posterior factorization
    q_ids = vcat(:Π,
                  expand([[pad(:M,k), pad(:W,k)] for k=1:rm_clusters]),
                  :Z, :Xc, :Ξ)

    # specify posterior factorization
    q = PosteriorFactorization(vars_e[:π], 
                               expand([[vars_e[pad(:m,k)], vars_e[pad(:w,k)]] for k=1:rm_clusters])...,
                               vars_e[:z], vars_e[:Xc], vars_e[:ξ], ids=q_ids)
    
    # return factorization
    return q
end


function perform_inference_online(g, y, t, z; nr_its=10)

    nr_clusters = g.n
    nr_samples = length(t)
    nr_freqs = g.d
    
    # calculate hyperparameter α based on cluster assignments of EM training
    ll = llpg(g, z_speech[:,2:end-1])
    llmin = maximum(ll, dims=2)
    llnorm = ll .- llmin
    znorm = exp.(llnorm) ./ sum(exp.(llnorm), dims=2)
    α_π_min = squeeze(sum(znorm, dims=1))
    
    # placeholders for priors
    μ_m_min = Array{Array{Float64,1},1}(undef, nr_clusters)
    nu_w_min = Array{Float64,1}(undef, nr_clusters)
    V_w_min = Array{Array{Float64,2},1}(undef, nr_clusters)

    # set priors (and current values later on)
    α_π_min = α_π_min
    p_z_min = exp.(ForneyLab.digamma.(α_π_min) .- ForneyLab.digamma.(sum(α_π_min))) ./ sum(exp.(ForneyLab.digamma.(α_π_min) .- ForneyLab.digamma.(sum(α_π_min))))
    for k = 1:nr_clusters
        μ_m_min[k] = g.μ[k,:]
        nu_w_min[k] = nr_freqs
        V_w_min[k] = diagm(1 ./g.Σ[k,:]) / nr_freqs
    end

    # memory placeholders
    z_mem = Array{Int64,1}(undef, nr_samples_train)

    # create progress bar
    p = Progress(nr_samples_train)

    # perform iterations
    for i = 1:nr_samples_train

        # create data dictionary
        data = Dict()
        data[:y] = y[i]
        for k = 1:nr_clusters
            data[pad(:μ_m,k)] = μ_m_min[k]
            data[pad(:nu_w,k)] = nu_w_min[k]
            data[pad(:V_w,k)] = V_w_min[k]
        end
        data[:C] = 1/bufsize*hcat(cos.(2*pi*fi*t[i]')', sin.(2*pi*fi*t[i]')')
        data[:α_π] = α_π_min

        # create marginals dictionary
        marginals = Dict()
        for k = 1:nr_clusters
            marginals[pad(:vars_m,k)] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=data[pad(:μ_m,k)], w=data[pad(:V_w,k)]*data[pad(:nu_w,k)])
            marginals[pad(:vars_w,k)] = ProbabilityDistribution(MatrixVariate, ForneyLab.Wishart, v=data[pad(:V_w,k)], nu=data[pad(:nu_w,k)])
        end
        marginals[:vars_π] = ProbabilityDistribution(Dirichlet, a=α_π_min)
        marginals[:vars_z] = ProbabilityDistribution(Categorical, p=p_z_min)
        marginals[:vars_Xc] = ProbabilityDistribution(Multivariate, ComplexNormal, μ=zeros(nr_freqs) .+ 0.0im, Γ=1e10*Ic(nr_freqs).+0.0im, C=mat(0.0+0.0im))
        marginals[:vars_ξ] = ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=z_speech[i, 2:end-1], v=1.0*diagm(squeeze(var(z_speech[:,2:end-1], dims=1))))

        for _ = 1:nr_its
            # perform updates
            Base.invokelatest(stepXc!, data, marginals)
            Base.invokelatest(stepΞ!, data, marginals)
            Base.invokelatest(stepZ!, data, marginals)
            Base.invokelatest(stepΠ!, data, marginals)
            for k = 1:nr_clusters
                Base.invokelatest(getfield(Main, Symbol("stepM_"*string(k,pad=2)*"!")), data, marginals)
                Base.invokelatest(getfield(Main, Symbol("stepW_"*string(k,pad=2)*"!")), data, marginals)
            end
        end

        # extract parameters and update beliefs
        α_π_min = marginals[:vars_π].params[:a]
        for k = 1:nr_clusters
            μ_m_min[k] = ForneyLab.unsafeMean(marginals[pad(:vars_m,k)])
            nu_w_min[k] = marginals[pad(:vars_w,k)].params[:nu]
            V_w_min[k] = marginals[pad(:vars_w,k)].params[:v]
        end
        z_mem[i] = findmax(marginals[:vars_z].params[:p])[2]
        p_z_min = exp.(ForneyLab.unsafeLogMean(marginals[:vars_π])) ./ sum(exp.(ForneyLab.unsafeLogMean(marginals[:vars_π])) )

        # update progress bar
        next!(p)

    end
end


# load packages
using Rocket, GraphPPL, ProgressMeter
using Distributions: Categorical, Dirichlet
using ReactiveMP

# export function
export update_model

# create model
@model [ default_factorisation = MeanField() ] function gaussian_mixture_model(nr_samples, nr_freqs, nr_mixtures, p_μ, p_γ, p_a)
    
    # dirichlet prior
    s ~ Dirichlet(p_a.a)
    
    # means and precisions
    m = randomvar(nr_mixtures, nr_freqs)
    w = randomvar(nr_mixtures, nr_freqs)
    for mix in 1:nr_mixtures
        for freq in 1:nr_freqs
            m[mix, freq] ~ NormalMeanPrecision(p_μ[mix].μ[freq], p_μ[mix].γ[freq])
            w[mix, freq] ~ GammaShapeRate(p_γ[mix].a[freq], p_γ[mix].b[freq])
        end
    end

    # selector variables
    z = randomvar(nr_samples)

    # observed signal
    y = datavar(Float64, nr_samples, nr_freqs)
    
    # create mixtures 
    for n in 1:nr_samples
        z[n] ~ Categorical(s)
        for freq in 1:nr_freqs
        #     tuple_a = tuple(m[:,freq]...)
        #     tuple_b = tuple(w[:,freq]...)
            y[n,freq] ~ NormalMixture(z[n], m[:,freq], w[:,freq])
        end
    end
    
    # specify scheduler
    scheduler = schedule_updates(m, w)
    
    return s, m, w, z, y, scheduler
end

# specify inference procedure
function update_model(model_name, data, priors, nr_files; nr_iterations=10)

    # extract priors
    (p_a, p_μ, p_γ) = priors

    # fetch dimensions
    (nr_frequencies, nr_samples) = size(data)
    nr_mixtures = length(p_μ)

    # filename
    filename = model_name*"_"*string(nr_frequencies)*"_"*string(nr_mixtures)*"_"*string(nr_files)*".h5"

    if isfile(filename)

        # load model 
        f = h5open(filename, "r")
        p_full_alpha = HDF5.read(f["p_full_alpha"], Float64);
        q_full_alpha = HDF5.read(f["q_full_alpha"], Float64);
        close(f)

        # create output
        p_full = Dirichlet(p_full_alpha)
        q_full = Dirichlet(q_full_alpha)

    # train model
    else

        # create model
        model, (s, m, w, z, y, scheduler) = gaussian_mixture_model(nr_samples, nr_frequencies, nr_mixtures, p_μ, p_γ, p_a, options=(limit_stack_depth=500,));
        
        # allocate space for marginals
        marg_switch = keep(Marginal)
        marg_selector = keep(Vector{Marginal})
        marg_m = keep(Matrix{Marginal})
        marg_w = keep(Matrix{Marginal})
        
        # allocate space for free energy
        fe = ScoreActor(Float64)
        
        # subscribe to marginals
        m_sub = subscribe!(getmarginals(m, IncludeAll()), marg_m)
        w_sub = subscribe!(getmarginals(w, IncludeAll()), marg_w)
        selector_sub = subscribe!(getmarginals(z, IncludeAll()), marg_selector)
        switch_sub = subscribe!(getmarginal(s, IncludeAll()), marg_switch)
        
        # subscribe to free energy
        fe_sub = subscribe!(score(Float64, BetheFreeEnergy(), model), fe)
        
        # initialize marginals
        setmarginal!(s, convert(Dirichlet, vague(Dirichlet, nr_mixtures)))
        for mix in 1: nr_mixtures
            for freq in 1:nr_frequencies
                setmarginal!(m[mix,freq], NormalMeanPrecision(p_μ[mix].μ[freq], p_μ[mix].γ[freq]))
                setmarginal!(w[mix,freq], GammaShapeRate(p_γ[mix].a[freq], p_γ[mix].b[freq]))
            end
        end
        
        # perform message passing
        ProgressMeter.@showprogress for it in 1:nr_iterations
            for n in 1:nr_samples
                for freq in 1:nr_frequencies
                    ReactiveMP.update!(y[n,freq], data[freq,n])
                end
            end
            release!(scheduler)
        end
        
        # unsubscribe
        unsubscribe!(fe_sub)
        unsubscribe!(switch_sub)
        unsubscribe!(m_sub)
        unsubscribe!(w_sub)

        @info "Model already adjusted."

        # write model 
        f = h5open(filename, "w")
        HDF5.write(f, "p_full_alpha", p_a.a);
        HDF5.write(f, "q_full_alpha", getvalues(marg_switch)[end].alpha);
        close(f)

        # create output
        p_full = Dirichlet(p_a.a)
        q_full = getvalues(marg_switch)[end]
        
    end

    # return extracted values
    return p_full, q_full

end
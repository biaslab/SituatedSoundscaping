using SpecialFunctions: loggamma, gamma

export model_reduction_all, model_reduction_steps, model_reduction_info

# helper functions for the model reduction
mvbeta(x::Array{Float64,1}) = prod(gamma.(x))/gamma(sum(x))
logmvbeta(x::Array{Float64,1}) = sum(loggamma.(x)) - loggamma(sum(x))
logmvbeta(x::Dirichlet) = sum(loggamma.(x.alpha)) - loggamma(sum(x.alpha))


# function for all-at-once model reduction
function model_reduction_all(p_full::Dirichlet, q_full::Dirichlet)

    # allocate array for approximate evidence differences
    Δp = Array{Float64,1}(undef, length(p_full))

    # calculate approximate evidence differences when removing individual components
    for k = 1:length(Δp)

        # create reduced prior
        p_reduced = deepcopy(p_full)
        p_reduced.alpha[k] = 0.001

        # calculate reduced approximate posterior
        q_reduced = Dirichlet(q_full.alpha + p_reduced.alpha - p_full.alpha)

        # calculate approximate difference in model evidence
        Δp[k] = logmvbeta(p_full) - logmvbeta(p_reduced) + logmvbeta(q_reduced) - logmvbeta(q_full)

    end

    # remove all terms where we gain an improvement in model evidence.
    p_a_reduced = deepcopy(p_full.alpha)
    p_a_reduced[Δp .> 0] .= 0.001

    # specify return values
    p_reduced = Dirichlet(p_a_reduced)
    q_reduced = Dirichlet(q_full.alpha + p_reduced.alpha - p_full.alpha)
    Δp = logmvbeta(p_full) - logmvbeta(p_reduced) + logmvbeta(q_reduced) - logmvbeta(q_full)

    # return values
    return p_reduced, q_reduced, Δp

end



# function for one-by-one model reduction
function model_reduction_steps(p_full::Dirichlet, q_full::Dirichlet)

    # allocate new prior
    p_reduced = deepcopy(p_full)

    # allocate array for approximate evidence differences
    Δp = Array{Float64,1}(undef, length(p_full))
    Δp .= 1
    
    # keep on going whilst there is an improvement to be made
    while maximum(Δp) > 0

        # reset values
        Δp .= -Inf
        
        # calculate approximate evidence differences when removing individual components
        for k = 1:length(Δp)

            # create reduced prior (skip if value is already reduced)
            p_reduced_tmp = deepcopy(p_reduced)
            if p_reduced_tmp.alpha[k] == 0.001
                continue
            else
                p_reduced_tmp.alpha[k] = 0.001
            end

            # calculate reduced approximate posterior
            q_reduced = Dirichlet(q_full.alpha + p_reduced_tmp.alpha - p_full.alpha)

            # calculate approximate difference in model evidence
            Δp[k] = logmvbeta(p_full) - logmvbeta(p_reduced_tmp) + logmvbeta(q_reduced) - logmvbeta(q_full)

        end

        if maximum(Δp) > 0
            p_reduced.alpha[Δp .== maximum(Δp)] .= 0.001
        end

    end

    # specify return values
    q_reduced = Dirichlet(q_full.alpha + p_reduced.alpha - p_full.alpha)
    Δp = logmvbeta(p_full) - logmvbeta(p_reduced) + logmvbeta(q_reduced) - logmvbeta(q_full)

    # return values
    return p_reduced, q_reduced, Δp

end


function model_reduction_info(p_full::Dirichlet, q_full::Dirichlet)

    # allocate new prior
    p_reduced = deepcopy(p_full)

    # allocate array for approximate evidence differences
    Δp = ones(length(p_full))
    
    # keep on going whilst there is an improvement to be made
    for k = 1:length(p_full)

        # reset values
        Δp .= -Inf
        
        # calculate approximate evidence differences when removing individual components
        for k = 1:length(Δp)

            # create reduced prior (skip if value is already reduced)
            p_reduced_tmp = deepcopy(p_reduced)
            if p_reduced_tmp.alpha[k] == 0.001
                continue
            else
                p_reduced_tmp.alpha[k] = 0.001
            end

            # calculate reduced approximate posterior
            q_reduced = Dirichlet(q_full.alpha + p_reduced_tmp.alpha - p_full.alpha)

            # calculate approximate difference in model evidence
            Δp[k] = logmvbeta(p_full) - logmvbeta(p_reduced_tmp) + logmvbeta(q_reduced) - logmvbeta(q_full)

        end

        p_reduced.alpha[Δp .== maximum(Δp)] .= 0.001

        @info "iteration "*string(k)*": Δp = "*string(maximum(Δp))

    end

end

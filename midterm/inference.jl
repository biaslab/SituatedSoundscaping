function perform_inference(spec_select, AR_order, q, iterations)
    algo = compatibility_fix(variationalAlgorithm(q))

    # evaluate algorithm
    eval(Meta.parse(algo))
    
    # priors
    current_a_γ = 1
    current_b_γ = 0.001
    current_Λ_θ = 1e-4 * Ic(AR_order)
    current_μ_θ = randn(AR_order)
    current_μ_Sprev = randn(AR_order)
    current_Λ_Sprev = 1e-4 * Ic(AR_order)
    F_tot = []
    a_γ_tot = []
    b_γ_tot = []
    μ_S_tot = []
    Λ_S_tot = []
    μ_θ_tot = []
    Λ_θ_tot = []
    μ_pred = Float64[]
    Λ_pred = Float64[]

    marginals = Dict()

    # create messages array 
    messages= Array{Message}(undef, 6)

    # loop through signal samples
    for sp in 1:length(spec_select)


        # update marginals
        marginals[:θ] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m=current_μ_θ, w=current_Λ_θ)
        marginals[:γ] = ProbabilityDistribution(ForneyLab.Univariate, ForneyLab.Gamma, a=current_a_γ, b=current_b_γ)
        marginals[:Sprev] = ProbabilityDistribution(ForneyLab.Multivariate, GaussianMeanPrecision, m=current_μ_Sprev, w=current_Λ_Sprev)

        # update data dictionary
        data = Dict(:y => spec_select[sp],
                    :μ_Sprev => current_μ_Sprev,
                    :Λ_Sprev => current_Λ_Sprev,
                    :μ_θ => current_μ_θ,
                    :Λ_θ => current_Λ_θ,
                    :a_γ => current_a_γ,
                    :b_γ => current_b_γ)

        # perform VMP iterations
        for it = 1:iterations

            # perform inference
             Base.invokelatest(stepS!, data, marginals, messages)
            if it == 1
                # perform 1 step prediction
                push!(μ_pred, ForneyLab.unsafeMean(messages[3].dist))
                push!(Λ_pred, ForneyLab.unsafePrecision(messages[3].dist))
            end
            Base.invokelatest(stepγ!, data, marginals, messages)
            Base.invokelatest(stepθ!, data, marginals, messages)

        end

        # fetch new parameters
        current_a_γ = marginals[:γ].params[:a]
        current_b_γ = marginals[:γ].params[:b]
        current_μ_S = ForneyLab.unsafeMean(marginals[:S])
        current_Λ_S = ForneyLab.unsafePrecision(marginals[:S])
        current_μ_θ = ForneyLab.unsafeMean(marginals[:θ])
        current_Λ_θ = ForneyLab.unsafePrecision(marginals[:θ])

        # save new parameters
        push!(a_γ_tot, current_a_γ)
        push!(b_γ_tot, current_b_γ)
        push!(μ_S_tot, current_μ_S)
        push!(Λ_S_tot, current_Λ_S)
        push!(μ_θ_tot, current_μ_θ)
        push!(Λ_θ_tot, current_Λ_θ)

        # update hidden state
        current_μ_Sprev = current_μ_S
        current_Λ_Sprev = current_Λ_S

    end
    
    return μ_pred, Λ_pred
end

function create_fg(AR_order, Λ_meas)
    # create factor graphs
    fg = FactorGraph()

    # AR node
    @RV γ ~ ForneyLab.Gamma(placeholder(:a_γ), placeholder(:b_γ))
    @RV θ ~ GaussianMeanPrecision(placeholder(:μ_θ, dims=(AR_order,)), placeholder(:Λ_θ, dims=(AR_order, AR_order)))
    @RV Sprev ~ GaussianMeanPrecision(placeholder(:μ_Sprev, dims=(AR_order,)), placeholder(:Λ_Sprev, dims=(AR_order, AR_order)))
    @RV S ~ LAR.Autoregressive(θ, Sprev, γ)

    # selection and input
    d = zeros(AR_order)
    d[1] = 1
    @RV x ~ DotProduct(d, S)
    @RV y ~ GaussianMeanPrecision(x, Λ_meas)
    placeholder(y, :y)
    
    q = RecognitionFactorization(θ, [S, Sprev], γ, ids=[:θ :S :γ])

    return q
end

function score_predictions(x_time, μ_pred, Λ_pred)
    
    # calculate MSE
    MSE = mean(abs2.(x_time - μ_pred))
    
    # calculate MAE
    MAE = mean(abs.(x_time - μ_pred))
    
    # calculate BME
    BME = sum([log.(pdf(Distributions.Normal(μ_pred[k], 1/sqrt(Λ_pred[k])), x_time[k])) for k = 1:length(x_time)])/length(x_time)
        
    # return metrics 
    return MSE, MAE, BME
    
end
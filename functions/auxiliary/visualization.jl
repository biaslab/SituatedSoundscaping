function plot_psd(ax, psd_mean, psd_std; color="orange", alpha=0.3)
    ax.plot(collect(0:(pi/(length(psd_mean)-1)-1e-10):pi), psd_mean, color=color, label="predicted psd")
    ax.fill_between(collect(0:(pi/(length(psd_mean)-1)-1e-10):pi), psd_mean .- psd_std, psd_mean .+ psd_std, color=color, alpha=alpha, zorder=100)
end

function summary_psd(θ, γ; freq_res=100, iterations=100)
    summary = [summary_psd_freq(θ, γ, f, freq_res=freq_res, iterations=iterations) for f=0:pi/freq_res-1e-10:pi]
    return [summary[k][1] for k = 1:length(summary)], [summary[k][2] for k = 1:length(summary)]
end

function summary_psd_freq(θ, γ, f; freq_res=100, iterations=100)
    # this function calculates the sampled mean and standard deviation of the power spectral density
       
    # create vector for dotproduct with AR coefficients
    z = exp.(-collect(1:length(mean(θ)))*1im.*f)
    
    # get samples of psd
    samples = [sample_psd_once(θ, γ, z) for _=1:iterations]
    
    # return mean and standard deviation
    return mean(samples), std(samples)
    
end

function sample_psd_once(θ, γ, z)
    # Get a single sample of the power spectral density
    
    # sample from AR coefficients
    sample_θ = rand(θ)
    
    # sample from process noise
    sample_γ = rand(γ)
    
    # return sampled value of distribution
    return 10*log10.((1/sample_γ)/abs2(1-sum(sample_θ.*z)))
end

function AR_distributions(μ_θ, Λ_θ, a_γ, b_γ)
    # convert parameters to distributions
    return MvNormalΛ(μ_θ, Λ_θ), Distributions.Gamma(a_γ, 1/b_γ)
end

function MvNormalΛ(μ, Λ)
    # convert precision matrix to covariance matrix for compatibility with Distributions.jl
    return Distributions.MvNormal(μ, Matrix(Hermitian(inv(Λ))))
end


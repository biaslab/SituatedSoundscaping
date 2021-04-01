export train_em

function process_data!(data, means, means_new, covs, covs_new, logπk, Nk, γ, it)
    d = zeros(maxsize(data)) 
    total_log_likelihood_propto = 0
    dsize = [1, 1]

    # loop through files
    @showprogress "iteration "*string(it)*" - " for k in 1:length(data)

        # load data
        f = h5open(data.list[k], "r")
        dsize .= size(f["data_logpower"])::Tuple{Int64, Int64}
        d[1:dsize[1], 1:dsize[2]] .= HDF5.readmmap(f["data_logpower"], Float64);

        # loop through observations
        for ki in 1:dsize[2]

            # set responsibilities to 0
            γ .= 0
            # loop through clusters
            for kii in 1:length(γ)
                # loop through frequencies
                for kiii in 1:dsize[1]
                    γ[kii] += lognormalvarpropto(d[kiii,ki], means[kiii,kii], covs[kiii,kii])
                end
            end

            # add log prior probability and normalize γ
            γ .+= logπk
            total_log_likelihood_propto += logsumexp(γ)
            softmax!(γ)

            # update mean and cov
            for kiii = 1:dsize[1]
                for kii = 1:length(γ)
                    means_new[kiii,kii] += d[kiii,ki] * γ[kii]
                    covs_new[kiii,kii] += (d[kiii,ki] - means[kiii,kii])^2 * γ[kii]
                end
            end

            # add to Nk
            Nk .+= γ
            
        end
        # close file
        close(f)

    end

    return total_log_likelihood_propto
end

# em step
function emstep!(data::Data, means::Array{Float64,2}, covs::Array{Float64,2}, πk::Array{Float64,1}, it::Int64)
    nr_frequencies, nr_mixtures = size(means)
    Nk = zeros(nr_mixtures)
    γ = zeros(nr_mixtures)
    logπk = log.(πk)
    means_new = zeros(size(means))
    covs_new = zeros(size(covs))

    total_log_likelihood_propto = process_data!(data, means, means_new, covs, covs_new, logπk, Nk, γ, it)
    total_log_likelihood_propto /= sum(Nk)

    # update statistics
    for ki = 1:nr_mixtures
        for k = 1:nr_frequencies
            means[k,ki] = means_new[k,ki] / Nk[ki]
            covs[k,ki] = covs_new[k,ki] / Nk[ki]
        end
    end
    normalize_sum!(Nk)
    πk .= Nk

    println("average proportional log-likelihood: ", total_log_likelihood_propto)
end

function train_em(model_name::String, data::Data, centers::Array{Float64,2}, πk::Array{Float64,1}; nr_iterations=10::Int64)

    # fetch dimensions 
    (nr_frequencies, nr_mixtures) = size(centers)
    nr_files = length(data)

    # filename
    filename = model_name*"_"*string(nr_frequencies)*"_"*string(nr_mixtures)*"_"*string(nr_files)*".h5"

    # check if model exists
    if isfile(filename)

        @info "EM model already trained."

        # load model 
        f = h5open(filename, "r")
        means = HDF5.read(f["means"], Float64);
        covs = HDF5.read(f["covs"], Float64);
        πk = HDF5.read(f["pi"], Float64);
        close(f)

    else

        # initialize clusters 
        means = copy(centers)
        covs = ones(size(means))
        πk = copy(πk)

        # perform EM iterations
        for it in 1:nr_iterations

            emstep!(data, means, covs, πk, it)

        end

        # save model
        f = h5open(filename, "w")
        HDF5.write(f, "means", means);
        HDF5.write(f, "covs", covs);
        HDF5.write(f, "pi", πk);
        close(f)
        
    end 

    return means, covs, πk

end
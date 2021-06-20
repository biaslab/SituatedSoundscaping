using ParallelKMeans

export train_kmeans

function train_kmeans(model_name::String, data::Data, nr_mixtures::Int64; power_dB=0::Real, nr_tries=10::Int64)

    # fetch dimensions 
    nr_frequencies = size(getindex(data, 1, "data_logpower"),1)
    nr_files = length(data)

    # filename
    filename = model_name*"_freq="*string(nr_frequencies)*"_mix="*string(nr_mixtures)*"_file="*string(nr_files)*"_power="*string(power_dB)*".h5"

    # check if model exists
    if isfile(filename)

        @info "Kmeans model already trained."

        # load model 
        f = h5open(filename, "r")
        centers = HDF5.read(f["centers"]);
        πk = HDF5.read(f["assignments"]);
        close(f)

    else

        # train model
        km = 0
        km_J = Inf
        for it = 1:nr_tries
            km_tmp = ParallelKMeans.kmeans(collect(data, "data_logpower"), nr_mixtures)
            if km_tmp.totalcost < km_J
                km = deepcopy(km_tmp)
                km_J = deepcopy(km_tmp.totalcost)
            end
        end
        # extract parameters
        centers = km.centers;
        πk = normalize_sum([count(x->x==k, km.assignments) for k in 1:nr_mixtures]);

        # save model
        f = h5open(filename, "w")
        HDF5.write(f, "centers", centers);
        HDF5.write(f, "assignments", πk);
        close(f)
        
    end 

    return centers, πk

end


function train_kmeans(model_name::String, data::Array{Float64,2}, nr_mixtures::Int64; power_dB=0::Real)

    # fetch dimensions 
    nr_frequencies = size(data,1)

    # filename
    filename = model_name*"_freq="*string(nr_frequencies)*"_mix="*string(nr_mixtures)*"_file=1_power="*string(power_dB)*".h5"

    # check if model exists
    if isfile(filename)

        @info "Kmeans model already trained."

        # load model 
        f = h5open(filename, "r")
        centers = HDF5.read(f["centers"]);
        πk = HDF5.read(f["assignments"]);
        close(f)

    else

        # train model
        km = ParallelKMeans.kmeans(data, nr_mixtures)
        
        # extract parameters
        centers = km.centers;
        πk = normalize_sum([count(x->x==k, km.assignments) for k in 1:nr_mixtures]);

        # save model
        f = h5open(filename, "w")
        HDF5.write(f, "centers", centers);
        HDF5.write(f, "assignments", πk);
        close(f)
        
    end 

    return centers, πk

end
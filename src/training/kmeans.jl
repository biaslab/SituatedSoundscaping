using ParallelKMeans

export train_kmeans

function train_kmeans(model_name::String, data::Data, nr_mixtures::Int64)

    # fetch dimensions 
    nr_frequencies = size(getindex(data, 1, "data_logpower"),1)
    nr_files = length(data)

    # filename
    filename = model_name*"_"*string(nr_frequencies)*"_"*string(nr_mixtures)*"_"*string(nr_files)*".h5"

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
        km = ParallelKMeans.kmeans(collect(data, "data_logpower"), nr_mixtures)
        
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
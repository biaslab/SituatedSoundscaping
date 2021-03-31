using FileIO
using DSP
using ProgressMeter
using HDF5

export prepare_data

"""
This function prepares the data for model training.
"""
function prepare_data(input_folder::String, output_folder::String; block_length::Int64=64, step_size::Int64=32, fs::Int64=16000, window::Function=hanning)

    # find files
    filenames = String[]
    rootpath = input_folder
    pattern  = ".flac"
    for (root, dirs, files) in walkdir(rootpath)
        for file in files
            if occursin(pattern, file) 
                # create string
                filename = string(root) * "/" * string(file)
    
                # replace slashes
                filename = replace(filename, "\\" => "/")
    
                # push file
                push!(filenames, filename)
            end
        end
    end
    
    # loop through files
    nr_files = length(filenames)
    output_files = String[]
    if !isdir(output_folder)
        mkdir(output_folder)
        ProgressMeter.@showprogress for (ind, file) in enumerate(filenames)

            # load data
            x_tmp, fs_tmp = load(file)

            # preprocess signal
            x_tmp = squeeze(x_tmp)
            @assert length(size(x_tmp)) == 1
            x_tmp = resample(x_tmp, fs/fs_tmp)
            x_tmp = x_tmp + 1e-5*randn(length(x_tmp))
            x_tmp = x_tmp .- mean(x_tmp)

            # convert to spectrum
            y_tmp = stft(x_tmp, block_length, block_length-step_size; onesided=true, fs=fs, window=window)
            y_tmp = transpose(y_tmp)
            y_tmp = log.(abs2.(y_tmp))
            y_tmp = collect(y_tmp')

            # save file
            append!(output_files, [output_folder*"/"*string(ind, pad=10)*".h5"])
            h5write(last(output_files), "data", y_tmp)
            h5write(last(output_files), "size", collect(size(y_tmp)))

        end

    else
        @info "data already processed."
        rootpath = output_folder
        pattern  = ".h5"
        for (root, dirs, files) in walkdir(rootpath)
            for file in files
                if occursin(pattern, file) 
                    # create string
                    filename = string(root) * "/" * string(file)
        
                    # replace slashes
                    filename = replace(filename, "\\" => "/")
        
                    # push file
                    push!(output_files, filename)
                end
            end
        end
    end

    # create data instance
    d = Data(output_files, Float64)

    return d

end

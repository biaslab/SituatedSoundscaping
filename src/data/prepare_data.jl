using FileIO
using DSP
using ProgressMeter
using HDF5

export prepare_data, read_recording, create_mixture_signal

function create_mixture_signal(filename_speech::String, filename_noise::String; duration_adapt::Int64=3, duration_test::Int64=10, fs::Int64=16000, block_length::Int64=64)

    # load and preprocess speech signal
    x_speech, fs_speech = load(filename_speech)
    x_speech = squeeze(x_speech)
    x_speech = resample(x_speech, fs/fs_speech)
    x_speech .+= 1e-5*randn(length(x_speech))
    x_speech .-= mean(x_speech)
    x_speech ./= std(x_speech)
    x_speech = x_speech[fs*duration_adapt+1:fs*(duration_adapt+duration_test)]

    # load and preprocess noise signal
    x_noise, fs_noise = load(filename_noise)
    x_noise = squeeze(x_noise)
    x_noise = resample(x_noise, fs/fs_noise)
    x_noise .+= 1e-5*randn(length(x_noise))
    x_noise .-= mean(x_noise)
    x_noise ./= std(x_noise)
    x_noise = x_noise[fs*duration_adapt+1:fs*(duration_adapt+duration_test)]

    # create mixture signal
    x_mixture = x_speech + x_noise

    # return signals
    return x_mixture, x_speech, x_noise

end

function read_recording(filename::String; duration::Real=3, fs::Int64=16000, block_length::Int64=64)
    X = h5read(filename, "data_complex")
    nr_observations = Int(floor(duration*fs/block_length))
    return X[:, 1:nr_observations]
end

"""
This function prepares the data for model training.
"""
function prepare_data(input_folder::String, output_folder::String; block_length::Int64=64, step_size::Int64=32, fs::Int64=16000, window::Function=hanning)

    # find files
    filenames = String[]
    rootpath = input_folder
    pattern1  = ".flac"
    pattern2  = ".wav"
    for (root, dirs, files) in walkdir(rootpath)
        for file in files
            if occursin(pattern1, file) || occursin(pattern2, file)
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

        # create filter bank
        filterbank = warped_filter_bank(block_duration_s=block_length/16000, nr_bands=Int(block_length/2)+1)

        ProgressMeter.@showprogress for (ind, file) in enumerate(filenames)

            
            # load data
            x_tmp, fs_tmp = load(file)

            # preprocess signal
            x_tmp = squeeze(x_tmp)
            @assert length(size(x_tmp)) == 1
            x_tmp = resample(x_tmp, fs/fs_tmp)
            x_tmp .+= 1e-5*randn(length(x_tmp))
            x_tmp .-= mean(x_tmp)
            x_tmp ./= std(x_tmp)

            # convert to spectrum
            Xs = Array{Complex{Float64},2}(undef, Int(block_length/2+1), Int(floor(length(x_tmp)/block_length)))
            ξs = Array{Float64,2}(undef, Int(block_length/2+1), Int(floor(length(x_tmp)/block_length)))
            for k = 1:Int(floor(length(x_tmp)/block_length))
                run!(filterbank, x_tmp[1+(k-1)*block_length:k*block_length])
                Xs[:,k] = get_frequency_coefficients(filterbank)
                ξs[:,k] = get_power(filterbank)
            end
            # y_tmp = stft(x_tmp, block_length, block_length-step_size; onesided=true, fs=fs, window=window)
            # y_tmp = log.(abs2.(y_tmp))

            # save file
            append!(output_files, [output_folder*"/"*string(ind, pad=10)*".h5"])
            h5write(last(output_files), "data_logpower", ξs)
            h5write(last(output_files), "data_complex", Xs)
            h5write(last(output_files), "size", collect(size(ξs)))

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

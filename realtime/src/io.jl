using WAV: wavread
using DSP: resample


function load_data(filenames::Array{String,1}; fs::Real=16000, duration::AbstractArray{T,1}=[1,1], levels_dB::AbstractArray{T,1}=zeros(length(filenames)), subtract_mean::Bool=true, normalize_std::Bool=true) where {T}

    # create array for data files
    y_train = Array{Array{Float64,1},1}(undef, length(filenames))
    y_test = Array{Array{Float64,1},1}(undef, length(filenames))

    # loop through different files
    for (index, filename) in enumerate(filenames)

        # load file (first load sampling frequency to prevent loading all data)
        _, fs_tmp = wavread(filename; subrange=1)
        fs_tmp = convert(Float64, fs_tmp)
        y_tmp, _ = wavread(filename; subrange=fs_tmp*maximum(duration))
        y_tmp = y_tmp[:,1]
        y_tmp = squeeze(y_tmp)

        # preprocess data
        if duration[1] > duration[2]
            y_train[index] = preprocess(y_tmp, fs_tmp; fs=fs, duration=duration[1], level_dB=levels_dB[index], subtract_mean=subtract_mean, normalize_std=normalize_std)
            y_test[index] = y_train[index][1:duration[2]*fs]
        else
            y_test[index] = preprocess(y_tmp, fs_tmp; fs=fs, duration=duration[2], level_dB=levels_dB[index], subtract_mean=subtract_mean, normalize_std=normalize_std)
            y_train[index] = y_test[index][1:duration[1]*fs]        
        end
    end

    return y_train, y_test

end
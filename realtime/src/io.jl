using WAV: wavread
using DSP: resample


function load_data(filenames::Array{String,1}; fs::Real=16000, duration::Real=1, levels_dB::Union{Array{Int64,1}, Array{Float64,1}}=zeros(length(filenames)), subtract_mean::Bool=true, normalize_std::Bool=true)

    # create array for data files
    y = Array{Array{Float64,1},1}(undef, length(filenames))

    ## loop through different files
    for (index, filename) in enumerate(filenames)

        ## load file (first load sampling frequency to prevent loading all data)
        _, fs_tmp = wavread(filename; subrange=1)
        fs_tmp = convert(Float64, fs_tmp)
        y_tmp, _ = wavread(filename; subrange=fs_tmp*duration)
        y_tmp = y_tmp[:,1]
        y_tmp = squeeze(y_tmp)

        ## preprocess data
        y[index] = preprocess(y_tmp, fs_tmp; fs=fs, duration=duration, level_dB=levels_dB[index], subtract_mean=subtract_mean, normalize_std=normalize_std)

    end

    return y

end
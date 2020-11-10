function export_results(x_sep::Array{Array{Float64,1},1}, x_true::Array{Array{Float64,1},1}, x_mix::Array{Float64,1}, X_sep::Array{Array{Complex{Float64},2},1}, X_true::Array{Array{Complex{Float64},2},1}, X_mix::Array{Complex{Float64},2}, gains::Array{T,1}, filenames::Array{String,1}; fs::Int64=16000) where {T}

    # create folder
    folder_path = "results/export"*join(["_"*split(split(filenames[k], ".")[1], "/")[2] for k = 1:length(filenames)])*"/"*Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    mkpath(folder_path)

    # get time axis
    t = collect(1:length(x_mix))/fs

    ## Plot 1: overview total signals

    # create axes for subplots
    _, ax = plt.subplots(ncols=1, nrows=length(filenames), figsize=(15, 5*(length(filenames))))


    # loop through rows and fill them
    for k = 1:length(filenames)
    
        # plot time-domain signal
        ax[k].plot(t, x_mix.+7, label="mixture")
        ax[k].plot(t, x_test[k], label="true")
        ax[k].plot(t, x_sep[k].-7, label="separated")
        ax[k].set_xlim([0, length(x_mix)/fs]), ax[k].set_ylim(-12, 12), ax[k].set_xlabel("time [sec]"), ax[k].set_ylabel("signal"), ax[k].grid(), ax[k].set_title(split(split(filenames[k], ".")[1], "/")[2]), ax[k].legend()
    
    end

    # save figure
    plt.gcf()
    plt.savefig(folder_path*"/signal_overview.png")


    ## Plot 2: Zoomed in version
        
    # create axes for subplots
    _, ax = plt.subplots(ncols=2, nrows=length(filenames), figsize=(15, 5*(length(filenames))))

    # loop through rows and fill them
    for k = 1:length(filenames)
        
        # plot time-domain signal
        ax[k,1].plot(t, x_test[k], label="true")
        ax[k,1].plot(t, x_sep[k], label="separated")
        ax[k,1].set_xlim([0, length(x_mix)/fs]), ax[k,1].set_xlabel("time [sec]"), ax[k,1].set_ylabel("signal"), ax[k,1].grid(), ax[k,1].set_title(split(split(filenames[k], ".")[1], "/")[2]), ax[k,1].legend()
        
        # plot zoomed-in version of signal
        ax[k,2].plot(t, x_test[k], label="true")
        ax[k,2].plot(t, x_sep[k], label="separated")
        ax[k,2].set_xlim([1.2, 1.21]), ax[k,2].set_xlabel("time [sec]"), ax[k,2].set_ylabel("signal"), ax[k,2].grid(), ax[k,2].set_title(split(split(filenames[k], ".")[1], "/")[2]), ax[k,2].legend()

    end

    # save figure
    plt.gcf()
    plt.savefig(folder_path*"/signal_zoom.png")


    ## Plot 3: Comparison spectrograms

    # create axes for subplots
    _, ax = plt.subplots(ncols=2, nrows=length(filenames), figsize=(15, 5*(length(filenames))))

    # loop through rows and fill them
    for k = 1:length(filenames)
        
        # plot true spectrogram
        ax[k,1].imshow(collect(transpose(log.(abs.(X_test[k])))), aspect="auto", origin="lower", cmap="jet", extent=[0, length(x_mix)/fs, 0, fs/2])
        ax[k,1].set_title("true: "*split(split(filenames[k], ".")[1], "/")[2])
        vrange = ax[k,1].get_images()[1].get_clim()
        ax[k,1].set_xlabel("time [sec]")
        ax[k,1].set_ylabel("Frequency [Hz]")
        
        # plot predicted spectrogram
        ax[k,2].imshow(collect(transpose(log.(abs.(X_sep[k])))), aspect="auto", origin="lower", cmap="jet", extent=[0, length(x_mix)/fs, 0, fs/2])
        ax[k,2].set_title("separated: "*split(split(filenames[k], ".")[1], "/")[2])
        ax[k,2].get_images()[1].set_clim(vrange)
        ax[k,2].set_xlabel("time [sec]")
        ax[k,2].set_ylabel("Frequency [Hz]")

    end

    # save figure
    plt.gcf()
    plt.savefig(folder_path*"/spectrograms.png")

    ## save audio filenames
    for k = 1:length(filenames)
        wavwrite(limit1(x_test[k]), folder_path*"/original_"*split(split(filenames[k], ".")[1], "/")[2]*string(gains[k])*"dB.wav", Fs=fs)
        wavwrite(limit1(x_sep[k]), folder_path*"/separated_"*split(split(filenames[k], ".")[1], "/")[2]*string(gains[k])*"dB.wav", Fs=fs)
    end

    wavwrite(limit1(x_test[1]), folder_path*"/original_"*join([split(split(filenames[k], ".")[1], "/")[2] for k = 1:length(filenames)])*string(gains[2])*"dB.wav", Fs=fs)
    wavwrite(limit1(x_sep[1]), folder_path*"/separated_"*join([split(split(filenames[k], ".")[1], "/")[2] for k = 1:length(filenames)])*string(gains[2])*"dB.wav", Fs=fs)
    wavwrite(limit1(x_mix), folder_path*"/mixture.wav", Fs=fs)
    wavwrite(limit1(x_mix), folder_path*"/noisy_"*join([split(split(filenames[k], ".")[1], "/")[2] for k = 1:length(filenames)])*string(gains[2])*"dB.wav", Fs=fs)

    ## copy setting from main.jl file
    cp("main.jl", folder_path*"/main.jl")

end
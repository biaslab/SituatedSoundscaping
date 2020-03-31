function plot_spectrogram(spec, fs; ax="none", fontsize=16, sparse=false)
    
    # create cartoon font
    # xkcd()
    
    # plot spectrogram
    if ax == "none"
        x = spec.power .- minimum(spec.power)
        if sparse 
            y = 1e-11
            for k = 1:100
                x = soft_thresholding2d(x, y);
            end
        end
        imshow(reverse(log10.(x), dims=1), 
               aspect="auto", 
               cmap="jet", 
               origin="lower",
               extent=[first(spec.time), last(spec.time), last(spec.freq), first(spec.freq)])
        
        # set limits of axes
        xlim(first(spec.time), last(spec.time))
        ylim(first(spec.freq), last(spec.freq))

        # set axes labels
        xlabel("time [sec]", fontsize=fontsize)
        ylabel("frequency [Hz]", fontsize=fontsize)
        
        # set ticks
        tick_params(labelsize=fontsize)
        
    else
        
        # creat plot
        x = spec.power .- minimum(spec.power)
        if sparse 
            y = 1e-11
            for k = 1:100
                x = soft_thresholding2d(x, y);
            end
        end
        ax.imshow(reverse(log10.(x), dims=1), 
               aspect="auto", 
               cmap="jet", 
               origin="lower",
               extent=[first(spec.time), last(spec.time), last(spec.freq), first(spec.freq)])        
        
        # set limits of axes
        ax.set_xlim(first(spec.time), last(spec.time))
        ax.set_ylim(first(spec.freq), last(spec.freq))

        # set axes labels
        ax.set_xlabel("time [sec]", fontsize=fontsize)
        ax.set_ylabel("frequency [Hz]", fontsize=fontsize)
        
        # set ticks
        ax.tick_params(labelsize=fontsize)
        
    end
    

end
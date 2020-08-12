## https://github.com/r9y9/SynthesisFilters.jl/blob/master/examples/AudioDisplay.jl

function inline_audioplayer(s::Array{Float64}, fs::Int ; returnmarkup=false)
    # This function creates an audio player with the sound signal `s`, with sample frequency `fs`
    
    # create IO buffer
    buf = IOBuffer()
    
    # write signal to buffer
    wavwrite(s, buf; Fs=fs)
    
    # encode the data stream
    @compat data = base64encode(buf.data)
    
    # create HTML audio player
    markup = """<audio controls="controls" {autoplay}>
                <source src="data:audio/wav;base64,$data" type="audio/wav" />
                Your browser does not support the audio element.
                </audio>"""
    
    if returnmarkup
        # return HTML markup
        return markup
    else
        # display audio player
        display(MIME("text/html"), markup)
    end
end

function audioplayers(s...; fs=16000::Int)
    # This function creates horizontally alligned audio players for (multiple) sound signals `s`, with sample frequency `fs`
    
    # create initial HTML markup string
    html_string = "<div>"
    
    # loop through sound signals
    for si in s
        # append HTML markup of audio player to html_string
        html_string = string(html_string, inline_audioplayer(si, fs, returnmarkup=true))
    end
    
    # close off string
    html_string = string(html_string, "</div>")
    
    # display audio players
    display(MIME("text/html"), html_string)

end
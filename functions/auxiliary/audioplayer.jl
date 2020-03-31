## https://github.com/r9y9/SynthesisFilters.jl/blob/master/examples/AudioDisplay.jl

function inline_audioplayer(s, fs)
    buf = IOBuffer()
    wavwrite(s, buf; Fs=fs)
    @compat data = base64encode(buf.data)
    markup = """<audio controls="controls" {autoplay}>
                <source src="data:audio/wav;base64,$data" type="audio/wav" />
                Your browser does not support the audio element.
                </audio>"""
    display(MIME("text/html") ,markup)
end
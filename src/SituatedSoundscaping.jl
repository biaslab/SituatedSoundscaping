__precompile__()

module SituatedSoundscaping

    using PyCall
    
    # present the package to the user
    @info "================================="
    @info "===   SITUATED SOUNDSCAPING   ==="
    @info "================================="

    # include helpers
    include("helpers.jl");

    # include custom distributions
    include("distributions.jl");

end # module


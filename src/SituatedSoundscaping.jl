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

    # include data object and data preparation
    include("data/data.jl")
    include("data/prepare_data.jl")

    # include warped filter bank
    include("warp.jl")

end # module


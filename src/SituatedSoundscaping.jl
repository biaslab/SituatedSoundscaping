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
    
    # include warped filter bank
    include("warp.jl")

    # include data object and data preparation
    include("data/data.jl")
    include("data/prepare_data.jl")

    # include training procedures
    include("training/kmeans.jl")
    include("training/em.jl")
    include("training/gs.jl")

    # include model reduction procedures
    include("adjustment/update_recording.jl")
    include("adjustment/bayesian_model_reduction.jl")
    include("adjustment/simplify.jl")

    # include source separation
    include("separation/source_separation.jl")
    include("separation/source_separation_gs.jl")
    include("separation/source_separation_gs_sum.jl")
    include("separation/source_separation_algonquin.jl")
    include("separation/source_separation_algonquin_paper.jl")
    include("separation/wiener.jl")

    # include metrics
    include("metrics/metrics.jl")

end # module


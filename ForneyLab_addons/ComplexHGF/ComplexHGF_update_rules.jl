import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate

export ruleVariationalComplexHGFOutNP,
       ruleVariationalComplexHGFIn1PN

function ruleVariationalComplexHGFOutNP(marg_X::Nothing, 
                                 marg_ξ::ProbabilityDistribution{Multivariate})
    
    # caluclate required mean
    mξ = unsafeMean(marg_ξ)

    # calculate required variance
    vξ = diag(unsafeCov(marg_ξ))

    # calculate new parameters
    mX = zeros(size(mξ)) .+ 0im
    vX = exp.(mξ - vξ/2) .+ 0im
    
    # create variational message
    return Message(Multivariate, ComplexNormal, μ=mX, Γ=diagm(vX), C=mat(0.0+0.0im))

end


function ruleVariationalComplexHGFIn1PN(marg_X::ProbabilityDistribution{Multivariate}, 
                                 marg_ξ::Nothing)
    
    # calculate required means
    mX = unsafeMean(marg_X)

    # calculate required variances
    vX = diag(unsafeCov(marg_X))

    # calculate new parameters
    mξ = log.(abs2.(mX) + real.(vX))
    vξ = 1.0*ones(length(mξ))

    # create variational message
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=mξ./vξ, w=diagm(1 ./ vξ))

end
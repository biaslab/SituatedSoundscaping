import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate

export ruleVariationalARFBOutNPPP,
       ruleVariationalARFBIn1PNPP,
       ruleVariationalARFBIn2PPNP,
       ruleVariationalARFBIn3PPPN


function ruleVariationalARFBOutNPPP(marg_y::Nothing, 
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    mθ = unsafeMean(marg_θ)
    mx = unsafeMean(marg_x)
    mw = unsafeMean(marg_w)
                        
    # calculate new parameters
    my = mθ .* mx
    wy = mw

    # create variational message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=wy*my, w=wy)

end


function ruleVariationalARFBIn1PNPP(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::Nothing, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})
    
    # caluclate required means
    my = unsafeMean(marg_y)
    mθ = unsafeMean(marg_θ)
    mw = unsafeMean(marg_w)

    # calculate required variances
    vθ = unsafeCov(marg_θ)

    # calculate new parameters
    wx = (vθ' + mθ*mθ') .* mw
    mx = inv(wx) * Diagonal(mθ) * mw * my

    # create variational message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=wx*mx, w=wx)

end


function ruleVariationalARFBIn2PPNP(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::Nothing, 
                                    marg_w::ProbabilityDistribution{MatrixVariate})

    # calculate required means
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mw = unsafeMean(marg_w)

    # calculate required variances
    vx = unsafeCov(marg_x)

    # calculate new parameters
    wθ = (vx' + mx*mx') .* mw
    mθ = inv(wθ) * Diagonal(mx) * mw * my

    # create variational message
    Message(Multivariate, GaussianWeightedMeanPrecision, xi=wθ*mθ, w=wθ)

end


function ruleVariationalARFBIn3PPPN(marg_y::ProbabilityDistribution{Multivariate}, 
                                    marg_x::ProbabilityDistribution{Multivariate}, 
                                    marg_θ::ProbabilityDistribution{Multivariate}, 
                                    marg_w::Nothing)

    # calculate required means
    my = unsafeMean(marg_y)
    mx = unsafeMean(marg_x)
    mθ = unsafeMean(marg_θ)

    # calculate required variances
    vy = unsafeCov(marg_y)
    vx = unsafeCov(marg_x)
    vθ = unsafeCov(marg_θ)

    # calculate new parameters
    v = vy + my*my' - (mθ .* mx)*my' - my*(mx .* mθ)' + Diagonal(mθ)*vx*Diagonal(mθ) + Diagonal(mx)*vθ*Diagonal(mx)  + (mθ .* mx)*(mθ .* mx)' + vθ.*vx
    nu = size(v,1) + 2 

    # create variational message
    Message(MatrixVariate, Wishart, v=inv(v), nu=nu)

end
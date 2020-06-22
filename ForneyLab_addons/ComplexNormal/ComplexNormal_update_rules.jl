import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate

export ruleSPComplexNormalOutNPPP,
       ruleSPComplexNormalIn1PNPP,
       ruleSPComplexNormalOutNCPP,
       ruleSPComplexNormalIn1CNPP,
       ruleVBComplexNormalOut,
       ruleVBComplexNormalIn1


ruleSPComplexNormalOutNPPP(msg_out::Nothing, 
                           msg_μ::Message{PointMass},
                           msg_Γ::Message{PointMass},
                           msg_C::Message{PointMass}) =
                           Message(Multivariate, ComplexNormal, μ=deepcopy(msg_μ.dist.params[:m]), Γ=deepcopy(msg_Γ.dist.params[:m]), C=deepcopy(msg_C.dist.params[:m]))


ruleSPComplexNormalIn1PNPP(msg_out::Message{PointMass}, 
                           msg_μ::Nothing,
                           msg_Γ::Message{PointMass},
                           msg_C::Message{PointMass}) =
                           Message(Multivariate, ComplexNormal, μ=deepcopy(msg_out.dist.params[:m]), Γ=deepcopy(msg_Γ.dist.params[:m]), C=deepcopy(msg_C.dist.params[:m]))


ruleSPComplexNormalOutNCPP(msg_out::Nothing, 
                           msg_μ::Message{ComplexNormal, Multivariate},
                           msg_Γ::Message{PointMass},
                           msg_C::Message{PointMass}) =
                           Message(Multivariate, ComplexNormal, μ=deepcopy(msg_μ.dist.params[:μ]), Γ=deepcopy(msg_Γ.dist.params[:m]) + unsafeCov(msg_μ.dist), C=deepcopy(msg_C.dist.params[:m]))


ruleSPComplexNormalIn1CNPP(msg_out::Message{ComplexNormal, Multivariate}, 
                           msg_μ::Nothing,
                           msg_Γ::Message{PointMass},
                           msg_C::Message{PointMass}) =
                           ruleSPComplexNormalOutNCPP(msg_mean, msg_out, msg_Γ, msg_C)


ruleVBComplexNormalOut(dist_out::Any,
                       dist_μ::ProbabilityDistribution{Multivariate},
                       dist_Γ::ProbabilityDistribution{MatrixVariate},
                       dist_C::ProbabilityDistribution{MatrixVariate}) =
                       Message(Multivariate, ComplexNormal, μ=unsafeMean(dist_μ), Γ=unsafeMean(dist_Γ), C=unsafeMean(dist_C))


ruleVBComplexNormalIn1(dist_out::ProbabilityDistribution{Multivariate},
                       dist_μ::Any,
                       dist_Γ::ProbabilityDistribution{MatrixVariate},
                       dist_C::ProbabilityDistribution{MatrixVariate}) =
                       Message(Multivariate, ComplexNormal, μ=unsafeMean(dist_out), Γ=unsafeMean(dist_Γ), C=unsafeMean(dist_C))
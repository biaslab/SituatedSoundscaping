using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!
import ForneyLab: unsafeMeanCov, unsafeCov, unsafeMean, unsafePrecision, Multivariate, MatrixVariate
    

"""
Description:

    The complex normal distribution.
    Note: in the derivations, C=0 is assumed for simplicity, at the cost of generality.

    f(out, μ, Γ, C) = 𝒩𝒞(out | μ, Γ, C)

Interfaces:
    1. out (output vector)
    2. μ (mean vector)
    3. Γ (covariance matrix)
    4. C (relation matrix)

Construction:
    ComplexNormal(out, μ, Γ, C, id=:some_id)

"""

# create random variable for vectors of gamma distributed RV's
mutable struct ComplexNormal <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function ComplexNormal(out, μ, Γ, C; id=ForneyLab.generateId(ComplexNormal))
        @ensureVariables(out, μ, Γ, C)
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        ForneyLab.addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:μ] = self.interfaces[2] = associate!(Interface(self), μ)
        self.i[:Γ] = self.interfaces[3] = associate!(Interface(self), Γ)
        self.i[:C] = self.interfaces[4] = associate!(Interface(self), C)

        return self
    end
end

slug(::Type{ComplexNormal}) = "𝒩𝒞"

format(dist::ProbabilityDistribution{Multivariate, ComplexNormal}) = "$(slug(ComplexNormal))(μ=$(format(dist.params[:μ])), Γ=$(format(dist.params[:Γ])), C=$(format(dist.params[:C])))"

ProbabilityDistribution(::Type{Multivariate}, ::Type{ComplexNormal}; μ::Array{Complex{Float64},1}, Γ::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}) = ProbabilityDistribution{Multivariate, ComplexNormal}(Dict(:μ=>μ, :Γ=>Γ, :C=>C))
ProbabilityDistribution(::Type{ComplexNormal}; μ::Array{Complex{Float64},1}, Γ::Array{Complex{Float64},2}, C::Array{Complex{Float64},2}) = ProbabilityDistribution{Multivariate, ComplexNormal}(Dict(:μ=>μ, :Γ=>Γ, :C=>C))
ProbabilityDistribution(::Type{ComplexNormal}; μ::Array{Complex{Float64},1}, Γ::Array{Float64,2}, C::Array{Complex{Float64},2}) = ProbabilityDistribution{Multivariate, ComplexNormal}(Dict(:μ=>μ, :Γ=>Γ.+0im, :C=>C))

function prod!( x::ProbabilityDistribution{Multivariate, ComplexNormal},
                y::ProbabilityDistribution{Multivariate, ComplexNormal},
                z::ProbabilityDistribution{Multivariate, ComplexNormal}=ProbabilityDistribution(Multivariate, ComplexNormal, μ=zeros(size(x.params[:μ])).+0im, Γ=(1e10+1e10im)*Ic(length(x.params[:μ])), C=mat(0.0+0.0im)))

    # TOO SIMPLIFIED (CASE FOR C=0 -> circular symmetry)
    z.params[:Γ] = inv(inv(x.params[:Γ]) + inv(y.params[:Γ])) .+ 0.0im
    z.params[:μ] = z.params[:Γ]*(inv(x.params[:Γ])*x.params[:μ] + inv(y.params[:Γ])*y.params[:μ]) .+ 0.0im
    z.params[:C] = mat(0.0+0.0im)
        
    return z
end

unsafeMean(dist::ProbabilityDistribution{Multivariate, ComplexNormal}) = dist.params[:μ] 
#ForneyLab.unsafeMean(dist::ProbabilityDistribution{Multivariate, ComplexNormal}) = dist.params[:μ] 

unsafeCov(dist::ProbabilityDistribution{Multivariate, ComplexNormal}) = dist.params[:Γ] 
#ForneyLab.unsafeCov(dist::ProbabilityDistribution{Multivariate, ComplexNormal}) = dist.params[:Γ]

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

@sumProductRule(:node_type     => ComplexNormal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Nothing, Message{PointMass}, Message{PointMass}, Message{PointMass}),
                :name          => SPComplexNormalOutNPPP)

@sumProductRule(:node_type     => ComplexNormal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Message{PointMass}, Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPComplexNormalIn1PNPP)

@sumProductRule(:node_type     => ComplexNormal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Nothing, Message{ComplexNormal}, Message{PointMass}, Message{PointMass}),
                :name          => SPComplexNormalOutNCPP)

@sumProductRule(:node_type     => ComplexNormal,
                :outbound_type => Message{ComplexNormal},
                :inbound_types => (Message{ComplexNormal}, Nothing, Message{PointMass}, Message{PointMass}),
                :name          => SPComplexNormalIn1CNPP)

@naiveVariationalRule(:node_type     => ComplexNormal,
                      :outbound_type => Message{ComplexNormal},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VBComplexNormalOut)

@naiveVariationalRule(:node_type     => ComplexNormal,
                      :outbound_type => Message{ComplexNormal},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VBComplexNormalIn1)
export simplify_model


function simplify_model(q_μ::Array{Dnormalvector,1}, q_γ::Array{Dgammavector,1}, p_reduced::Dirichlet)

    # check which indices have not been reduced
    idx_remaining = p_reduced.alpha .> 0.001

    # simplify models 
    q_μ = q_μ[idx_remaining]
    q_γ = q_γ[idx_remaining]
    q_a = Ddirichlet(p_reduced.alpha[idx_remaining])

    # return statistics
    return q_μ, q_γ, q_a

end
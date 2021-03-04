

function updateLinearPredictor!(Xβ::Vector{T},
                                X::Matrix{T},
                                X_mean_scaled::Vector{T},
                                β::Vector{T},
                                indices,
                                standardize::Bool) where {T<:Float64}
    Xβ[1:end] = X[:, indices]*β[indices]
end

function updateLinearPredictor!(Xβ::Vector{T},
                                X::SparseMatrixCSC{T},
                                X_mean_scaled::Vector{T},
                                β::Vector{T},
                                indices,
                                standardize::Bool) where {T<:Float64}
    Xβ[1:end] = X[:, indices]*β[indices]

    if standardize
        Xβ[1:end] .-= dot(β[indices], X_mean_scaled[indices])
    end
end
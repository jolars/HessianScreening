using Distributions
using SparseArrays
using Base.Threads

function updateCorrelation!(c::Vector{T},
                            X::SparseMatrixCSC{T},
                            X_mean_scaled::Vector{T},
                            residual::Vector{T},
                            indices,
                            standardize::Bool) where {T <: Float64}
    residual_sum = sum(residual)

    @inbounds @threads for j in indices
        c[j] = dot(X[:, j], residual)

        if standardize
            c[j] -= X_mean_scaled[j]*residual_sum
        end
    end
end

function updateCorrelation!(c::Vector{T},
                            X::Matrix{T},
                            X_mean_scaled::Vector{T},
                            residual::Vector{T},
                            indices,
                            standardize::Bool) where {T <: Float64}
    @inbounds @threads for j in indices
        c[j] = dot(X[:, j], residual)
    end
end
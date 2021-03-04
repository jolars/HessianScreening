using Statistics
using Base.Threads

function standardizeX!(X::Matrix{Float64}, standardize::Bool = true)
    n, p = size(X)

    X_mean = zeros(p)
    X_sd = ones(p)

    if standardize
        @inbounds @threads for j in 1:p
            X_mean[j] = mean(X[:, j])
            X[:, j] .-= X_mean[j]

            X_sd_j = std(X[:, j], corrected = false)

            if X_sd_j != 0.0
                X_sd[j] = X_sd_j
                X[:, j] ./= X_sd[j]
            else
                X_sd[j] = 1.0
            end
        end
    end

    X_mean, X_sd
end

function standardizeX!(X::SparseMatrixCSC{Float64}, standardize::Bool = true)
    n, p = size(X)

    X_mean = zeros(p)
    X_sd = ones(p)

    if standardize
        @inbounds @threads for j in 1:p
            X_mean[j] = mean(X[:, j])

            X_sd_j = std(X[:, j], corrected = false)

            if X_sd_j != 0.0
                X_sd[j] = std(X[:, j], corrected = false)
                X[:, j] ./= X_sd[j]
            else
                X_sd[j] = 1.0
            end
        end
    end

    X_mean, X_sd
end

function standardizeY!(::Normal, y::Vector{Float64})
    y .-= mean(y)
end

function standardizeY!(::Binomial, y::Vector{Float64})
end
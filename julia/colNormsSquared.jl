using Base.Threads

function colNormsSquared(X::AbstractMatrix{Float64})
    n, p = size(X)
    out = Vector{Float64}(undef, p)

    @inbounds @threads for j in 1:p
        out[j] = norm(X[:, j])^2
    end

    out
end
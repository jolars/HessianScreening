using Random
using LinearAlgebra
using SparseArrays

function generateDesign(n,
                        p;
                        family = Normal(),
                        density = 1,
                        ρ = 0,
                        s = 5,
                        ρtype = "constant",
                        βtype = 1,
                        snr = 2)

    s = min(s, p)
    β = zeros(p)

    if density != 1 && ρ > 0
        throw(DomainError(ρ, "when density != 1, ρ must be 0"))
    end

    if βtype == 1
        ind = Int.(round.(range(1, p, length = s)))
        β[ind] .= 1
    elseif βtype == 2
        β[1:s] .= 1
    elseif βtype == 3
        β[1:s] .= range(10, 0.5, length = s)
    elseif βtype == 4
        β[1:s] .= 1
        β[(s + 1):p] = 0.5.^(1:(p - s))
    end

    if density == 1
        X = randn(n, p)
    else
        X = sprand(n, p, density)
    end

    if ρ != 0
        if ρtype == "auto"
            # autocorrelated predictors
            inds = 1:p
            Σ = Symmetric(ρ.^abs.(inds .- inds'))
            chol = cholesky(Σ)

            X = X * chol.L
            σ = √((β' * Σ * β)/snr)

        elseif ρtype == "constant"

            X = repeat(randn(n, 1), 1, p)*sqrt(ρ) + sqrt(1 - ρ)*randn(n, p)
            σ = √((ρ * sum(β)^2 + (1 - ρ) * sum(β.^2))/snr)
            #X = X * S
        else
            throw(DomainError(ρtype, "must be 'auto' or 'constant'"))
        end
    else
        Σ = I(p)
        σ = √((β' * Σ * β)/snr)
    end

    y = X*β + σ*randn(n)

    if family isa Binomial
        y = (sign.(y) .+ 1) ./ 2
    end

    X, y
end

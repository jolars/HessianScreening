using Statistics
using GLMNet
using Random
using LinearAlgebra
using Plots

include("updateHessianInverse.jl")

Random.seed!(15)

n = 100
p = 3
N = max(n, p)

X = rand(Float64, (n, p))
β = rand(Float64, p)
y = X*β + rand(Float64, n)

# standardize
for j = 1:p
    X[:,j] = (X[:,j] .- mean(X[:,j]))/std(X[:,j])
end

y = y .- mean(y)

function homotopyLasso(X, y)
    n, p = size(X)

    c = X'y

    active = falses(p)
    active[argmax(abs.(c))] = true
    active_prev = copy(active)
    inactive = .!active

    λ = maximum(c)
    λs = []

    β = zeros(p)
    βs =  Array{Float64}(undef, p, 0)

    # sign of beta
    s = zeros(p)
    s[active] = sign.(c[active])

    # initialize hessian and hessian inverse
    H  = X[:, active]'X[:, active]
    H⁻ = inv(H)

    keep_going = true

    while true
        active_prev = copy(active)

        X₁ = X[:, active]

        # s: sign of beta == sign of gradient (negative sign of inverse gradient)
        s[active] = sign.(c[active])

        # update solution
        β[active] = H⁻ * (X₁'y - s[active])
        #β[active] = Symmetric(X₁'X₁) \ (X₁'y .- λ * s[active])

        global βs = [βs β]
        global λs = [λs; λ]

        if (sum(active) >= p)
            break
        end

        # compute gradient of c (X'(y - Xβ)) with respect to lambda
        ∇c = X'X₁ * H⁻ * s[active]

        # find lambda at next point when predictor leaves model
        a = (c[inactive] - λ .* ∇c[inactive]) ./ (1.0 .- ∇c[inactive])
        b = (λ .* ∇c[inactive] .- c[inactive]) ./ (1.0 .+ ∇c[inactive])

        λ_next_nonzero = maximum(max(a, b))

        # update inverse gradient
        global c = c .+ (λ - λ_next)*∇c

        inactive_set = findall(inactive)
        next_active = inactive_set[argmax(max(a, b))]
        active[next_active] = true
        global inactive = .!active

        global λ = copy(λ_next)

        H, H⁻ = updateHessianInverse(H, H⁻, X, active_prev, active)
    end

    if (p <= n)
        β = Symmetric(X'X) \ X'y
        βs = [βs β]
        λs = [λs; 0.0]
    end

    (βs, λs)
end

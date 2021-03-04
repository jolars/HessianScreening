# soft-thresholding (lasso prox)
function prox(x, λ)
    sign(x) * max(abs(x) - λ, 0.0)
end

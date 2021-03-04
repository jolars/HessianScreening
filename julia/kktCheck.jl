using Base.Threads

# check the kkt conditions
function kktCheck!(violations, c, λ, screened, check_set)
    @inbounds @threads for j in check_set
        if abs(c[j]) >= λ
            violations[j] = true
            screened[j] = true
        end
    end
end

using SparseArrays

# read and format a dataset stored in the sparse format from LIBSVM
function readLIBSVM(data)
        
    l = split.(data, r"[ ]+")

    y = nothing
    n = length(l)

    if !(occursin(":", l[1][1]))
        y = Float64[]
        for i in 1:n
            push!(y, parse(Float64, popfirst!(l[i])))
        end
    end

    row_ind = Int64[]
    col_ind = Int64[]
    val = Float64[]

    for i in 1:n
        if l[i][1] != ""
            x = split.(l[i], r":")         
            
            for j in 1:length(x)
                if x[j][1] != ""
                    push!(row_ind, i)
                    push!(col_ind, parse(Int64, x[j][1]))
                    push!(val, parse(Float64, x[j][2]))
                end
            end
        end
    end

    X = SparseArrays.sparse(row_ind, col_ind, val)

    (X = X, y = y) 
end
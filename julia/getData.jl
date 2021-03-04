using SparseArrays
using FileIO
using CodecBzip2
using DrWatson

include("readLIBSVM.jl")

function sparsity(A)
    1 - nnz(A)/length(A)
end

function getlibsvmdata(url)
    if occursin(r"\.bz2$", url)
        file = download(url, tempname() * ".bz2")
        tmp = String(transcode(Bzip2Decompressor, read(file)))
        tmp = split(tmp, r"\n")
        deleteat!(tmp, length(tmp))
    else
        file = download(url)
        tmp = readlines(file)
    end

    X, y = readLIBSVM(tmp)

    if sparsity(X) < 0.8
        X = Array(X)
    end

    uvals = sort(unique(y))

    if length(uvals) == 2
        y[y .== uvals[1]] .= 0
        y[y .== uvals[2]] .= 1
    end

    X, y
end

function downloadDatasets()
    datafiles = Dict(
        "colon-cancer" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/colon-cancer.bz2",
        "covtype" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2",
        "diabetes" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes",
        "duke-breast-cancer" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/duke.bz2",
        "e2006-log1p-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.test.bz2",
        "e2006-log1p-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/log1p.E2006.train.bz2",
        "e2006-tfidf-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.test.bz2",
        "e2006-tfidf-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/E2006.train.bz2",
        #"epsilon-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
        #"epsilon-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
        "german-numer" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/german.numer",
        "gisette-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2",
        "gisette-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2",
        "heart" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/heart",
        "ijcnn1-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2",
        "ijcnn1-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",
        "ionosphere" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale",
        "leukemia-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.t.bz2",
        "leukemia-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/leu.bz2",
        "madelon-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon.t",
        "madelon-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/madelon",
        "mushrooms" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/mushrooms",
        "news20" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "phishing" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing",
        "pyrim-scaled-expanded5" => "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/QJEUKR/GVA3LP",
        "rcv1-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",
        "real-sim" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "sonar" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/sonar_scale",
        "splice-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice",
        "splice-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/splice.t",
        "triazines-scaled-expanded4" => "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/QJEUKR/L081UH",
        "YearPredictionMSD-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2",
        "YearPredictionMSD-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2",
        # "epsilon-test" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2",
        # "epsilon-train" => "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2",
        # NOTE(jolars): epsilon datasets are very large
    )

    i = 0

    for (key, value) in datafiles
        i += 1
        print("$(i)/$(length(datafiles)): $(key)..")
        filename = joinpath(datadir(), key * ".jld")
        if !isfile(filename)
            print("downloading..")

            X, y = getlibsvmdata(value)

            # remove zero-variance predictors
            v = [var(X[:, j]) for j in 1:size(X, 2)]
            X = X[:, v .!= 0]

            FileIO.save(filename, "X", X, "y", y)
            println("done!")
        else
            println("..already downloaded, skipping!")
        end
    end
end

function getData(dataset::AbstractString)
    filename = joinpath(datadir(), dataset * ".jld")

    if isfile(filename)
        data = load(filename)
        X = data["X"]
        y = data["y"]

        return X, y
    else
        throw(DomainError(dataset, "dataset is not supported or cannot be found"))
    end
end
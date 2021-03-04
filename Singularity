Bootstrap: docker
from: r-base:4.0.4

%files
    data /Project/data
    results /Project/results
    experiments /Project/experiments
    renv /Project/renv
    renv.lock /Project/renv.lock
    .Rprofile /Project/.Rprofile

%post
    cd Project

    Rscript -e 'renv::restore()'

    chmod -R a+rX /Project

%runscript
    if [ -z "$@" ]; then
        # if theres none, test R
        Rscript -e 'sessionInfo()'
    else
        # if theres an argument, then run it and hope it's an R script
        cd /Project
        Rscript -e "source(\"experiments/$@\")"
    fi
Bootstrap: docker
from: rocker/r-ver:4.0.4

%files
    data /Project/data
    results /Project/results
    experiments /Project/experiments
    renv /Project/renv
    renv.lock /Project/renv.lock
    HessianScreening /Project/HessianScreening
    .Rprofile /Project/.Rprofile

%post
    # need to switch from pthreads to openmp to get right performance
    apt-get update
    apt-get install -y libopenblas-openmp-dev libopenblas0-openmp
    apt-get remove -y libopenblas-pthread-dev libopenblas0-pthread

    cd Project

    Rscript -e 'renv::restore()'

    R CMD INSTALL --preclean --no-multiarch HessianScreening

    chmod -R a+rX /Project

%runscript
    if [ -z "$@" ]; then
        # if there's no argument, simply test R
        Rscript -e 'sessionInfo()'
    else
        # if there's an argument, then run it and hope it's an R script
        cd /Project
        Rscript -e "source(\"experiments/$@\")"
    fi
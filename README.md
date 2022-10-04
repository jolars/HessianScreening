# Code for the Hessian Screening Rule

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2104.13026)

## Results

The results from the simulations, which were run on a dedicated HPC
cluster, are stored in the [results folder](results/). The figures and
tables in the paper, generated from these results, are stored in
[`figures/`](figures/) and [`tables/`](tables/) respectively.

## Reproducing the Results

The results from our paper were run through a singularity container.
Check the releases for pre-built singularity containers that you can
download and use.

To reproduce the results, **always** use the singularity container. To
run an experiment from the singularity container, call

``` shell
singularity run --bind results:/project/results container.sif <script>
```

where `<script>` should be a name of a script in the [experiments
folder](experiments/), such as `simulateddata.R`.

### Re-building the Singularity Container

If you want to re-build the singularity container from scratch (or
simply want to clone the repo to your local drive), you can do so via
the following steps.

1.  Make sure you have installed and enabled
    [Git-LFS](https://git-lfs.github.com/). On ubuntu, for instance, you
    can install Git-LFS by calling
    
    ``` shell
    sudo apt update
    sudo apt install git-lfs
    ```
    
    Then activate git-lfs by calling
    
    ``` shell
    git lfs install
    ```

2.  Clone the repository to your local hard drive. On linux, using SSH
    authentication, run
    
    ``` shell
    git clone git@github.com:jolars/HessianScreening.git
    ```

3.  Navigate to the root of the repo and build the singularity container
    by calling
    
    ``` shell
    cd HessianScreening
    sudo singularity build container.sif Singularity
    ```

Then proceed as in [Reproducing the Results](#reproducing-the-results)
to run the experiments.

### Running Experiments without Singularity (Not Recommended\!)

Alternatively, you may also reproduce the results by cloning this
repository, then either opening the `HessianScreening.Rproj` file in R
Studio or starting R in the root directory of this folder (which will
activate the renv repository) and then run

``` r
renv::restore()
```

to restore the project library. Then build the R package (see below) and
run the simulations directly by running the scripts in the experiments
folder. This is **not recommended**, however, since it, unlike the
Singularity container approach, does not exactly reproduce the software
environment used when these simulations where originally run and may
result in discrepancies due to differences in for instance operating
systems, compilers, and BLAS/LAPACK implementations.

## R Package

If you want to build and experiment with the package, you can do so by
calling

``` shell
 R CMD INSTALL  .
```

provided you have `cd`ed to the root folder of this repository. First
ensure, however, that you have enabled the renv project library by
calling `renv::restore()` (see the section above).

## Data

The datasets used in these simulations are stored in the [data
folder](data/). Scripts to retrieve these datasets from their original
sources can be found in [`data-raw/`](data-raw/).

## Forking and Git-LFS

Note that pushing large files using Git-LFS against forks of this repo
[counts against the bandwidth limits of this
repo](https://docs.github.com/en/github/managing-large-files/collaboration-with-git-large-file-storage),
and so may fail if these limits are exceeded. If you for some reason
need to do this and it fails, please file as issue here.

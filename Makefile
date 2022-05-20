DELETE  := rm -rf
PKGNAME := $(shell sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION)
PKGVERS := $(shell sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION)
PKGSRC  := $(shell basename `pwd`)
FIGDIR  := ./figures
RESULTDIR   := ./results
FIG_SCRIPTS := $(wildcard figures/*.R)
FIG_OUTPUTS := $(FIG_SCRIPTS:.R=.pdf)
TABLE_SCRIPTS := $(wildcard tables/*.R)
TABLE_OUTPUTS := $(wildcard tables/*.R)

all: install

clean:
	$(DELETE) src/*.o src/*.so

clean-output: clean-tables clean-figures
	$(DELETE) figures/*.pdf figures/*.tex

clean-figures:
	$(DELETE) figures/*.pdf figures/*.tex

clean-tables:
	$(DELETE) tables/*.csv tables/*.csv

document: 
	Rscript -e 'devtools::document(roclets = c("rd", "collate", "namespace"))'

compile-attributes: 
	Rscript -e 'Rcpp::compileAttributes()'

build: document compile-attributes
	cd ..;\
	R CMD build --no-manual $(PKGSRC) --no-build-vignettes

build-cran: compile-attributes
	cd ..;\
	R CMD build $(PKGSRC)

install: compile-attributes
	R CMD INSTALL --preclean --no-multiarch --with-keep.source .

check: compile-attributes
	Rscript -e 'devtools::check()'

test: compile-attributes
	Rscript -e 'devtools::test()'

vignettes:
	Rscript -e 'devtools::build_vignettes()'

container:
	$(DELETE) container.sif;\
	sudo singularity build container.sif Singularity

figs: $(FIG_OUTPUTS)

$(FIG_OUTPUTS): figures/%.pdf: figures/%.R
	echo $<
	Rscript $<

tabs: 
	Rscript tables/realdata.R
	Rscript tables/violations.R

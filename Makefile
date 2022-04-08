DELETE  := rm -rf
PKGNAME := $(shell sed -n "s/Package: *\([^ ]*\)/\1/p" DESCRIPTION)
PKGVERS := $(shell sed -n "s/Version: *\([^ ]*\)/\1/p" DESCRIPTION)
PKGSRC  := $(shell basename `pwd`)

all: install

clean:
	$(DELETE) src/*.o src/*.so

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



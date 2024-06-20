# data from Patrick Breheny's webpage

# scheetz2006
tmpfile <- tempfile(fileext = ".rds")
download.file("https://github.com/IowaBiostat/data-sets/raw/main/Scheetz2006/Scheetz2006.rds", tmpfile)
scheetz <- readRDS(tmpfile)
saveRDS(scheetz, file.path("data", "scheetz.rds"))

# bcTCGA
tmpfile <- tempfile(fileext = ".rds")
download.file("https://github.com/IowaBiostat/data-sets/raw/main/brca1/brca1.rds", tmpfile)
bc_tcga <- readRDS(tmpfile)
saveRDS(bc_tcga, file.path("data", "bc_tcga.rds"))

library(Matrix)

data <- tempfile()
labels <- tempfile()

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.data",
  data
)

d <- scan(data, sep = " ")

x <- matrix(d, nrow = 100, byrow = TRUE)[, 1:10000]
x <- Matrix(x, sparse = TRUE)

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/arcene/ARCENE/arcene_train.labels",
  labels
)

y <- scan(labels, sep = " ")
y <- (y + 1) / 2

arcene <- list(X = x, y = y)

saveRDS(arcene, "data/arcene.rds")

unlink(labels)
unlink(data)

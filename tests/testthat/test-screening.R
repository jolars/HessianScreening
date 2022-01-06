# test_that("screening methods work", {
#   set.seed(1014)

#   n <- 50
#   p <- 200

#   tol_gap <- 1e-4

#   screening_types <- c("hessian", "working", "strong")

#   for (density in c(1, 0.5)) {
#     for (family in c("gaussian", "binomial")) {
#       d <- generateDesign(n, p, family = family)

#       X <- d$X
#       y <- d$y

#       for (screening_type in screening_types) {
#         if (family == "binomial" && screening_type == "edpp") {
#           next
#         }

#         fit <- lassoPath(
#           X,
#           y,
#           family = family,
#           screening_type = screening_type,
#           # force_kkt_check = TRUE,
#           tol_gap = tol_gap
#         )

#         # if (screening_type %in% c("gap_safe")) {
#         #   expect_equal(sum(fit$violations), 0)
#         # }
#       }
#     }
#   }
# })

#!/usr/bin/env Rscript --vanilla

remove_extension <- function(file) {
    sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(file))
}


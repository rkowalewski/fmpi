#!/usr/bin/env Rscript

# suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
#suppressMessages(library(gridExtra))
#suppressMessages(library(grid))
suppressMessages(library(argparser))
suppressMessages(library(ggsci))



thisFile <- function() {
        cmdArgs <- commandArgs(trailingOnly = FALSE)
        needle <- "--file="
        match <- grep(needle, cmdArgs)
        if (length(match) > 0) {
                # Rscript
                return(normalizePath(sub(needle, "", cmdArgs[match])))
        } else {
                # 'source'd via R console
                return(normalizePath(sys.frames()[[1]]$ofile))
        }
}

# load util functions
source(paste0(dirname(thisFile()), "/internal/", "utils.R"))

params <- arg_parser("visualizing mergebench performance")
params <- add_argument(params, "--input", help = "input file")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")
params <- add_argument(params, "--title", help = "PDF Title", default = "")
params <- add_argument(params, "--caption", help = "PDF Caption", default = "")
params <- add_argument(params, "--paper", help = "paper format (see R pdf manual)", default = "special")

argv <- parse_args(params)


data.in <- read.csv(file=argv$input, header=TRUE, sep=",")


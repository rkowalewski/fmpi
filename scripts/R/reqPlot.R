#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(argparser))

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

params <- arg_parser("visualizing comm progress efficiency")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")

argv <- parse_args(params)


df.in <- 0

if (argv$input == "-") {
    df.in <- read.csv(file('stdin'), header=TRUE)
} else {
    df.in <- read.csv(file=argv$input, header=TRUE)
}

Win <- function(Algo) {
    nreqs <- gsub("([A-Za-z]+)([0-9]+$)", "\\2", Algo)
    strtoi(nreqs)
}

# We want only the Waitsome algorithms to visualize
filterPat <- ".*Waitsome.*"

df.in <- df.in %>%
    filter(Measurement == "Ncomm_rounds" & grepl(filterPat, Algo)) %>%
    mutate(PPN = Procs / Nodes,
           ideal = ceiling((Procs - 1) / Win(Algo)),
           efficiency = ideal / median)

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1)  # move them .05 to the left and right

mylimit <- function(x) {
    limits <- c(min(x) - .1, max(x) + .1)
    limits
}


pdf(argv$output,paper="a4")

p = ggplot(data = df.in) +
    geom_line(
        aes(x = factor(Nodes), y = efficiency, colour = Algo, group = Algo), position = pd) +
    scale_y_continuous(limits = mylimit) +
    scale_colour_brewer(type="qal", palette="Paired") +
    geom_point(
        aes(x = factor(Nodes), y = efficiency, colour = Algo, group = Algo, shape=Algo),
        position = pd, size = 2) +
    theme_bw() + xlab('Nodes') + ylab('Comm_Rounds') +
    facet_wrap_paginate(~Blocksize, ncol=1, nrow=3, scales="free_y", page=NULL)

# Here we add our special class
if(!exists("print.gg_multiple", mode="function")) {
    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
}
class(p) <- c('gg_multiple', class(p))

print(p)

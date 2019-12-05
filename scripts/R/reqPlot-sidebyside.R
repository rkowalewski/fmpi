#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(argparser))
suppressMessages(library(cowplot))

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
params <- add_argument(params, "--title", help = "title of plot", default = "")
params <- add_argument(params, "--paper", help = "paper format", default = "special")

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

# We want only the Testsome / Waitsome algorithms to visualize
filterPat <- ".*some.*"

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

mylimit2 <- function(x) {
    limits <- c(min(x), max(x))
    limits
}


pdf(argv$output, paper=argv$paper)

p <- ggplot(data = df.in, aes(x = factor(Nodes), y = efficiency, colour = Algo, group = Algo)) +
    geom_line(
        position = pd) +
    scale_y_continuous(limits = mylimit) +
    scale_colour_brewer(type="qal", palette="Paired") +
    geom_point(
        aes(shape=Algo),
        position = pd, size = 2) +
    xlab('Nodes') + ylab('Progress Efficiency') +
    theme_bw()

if (argv$title != '') {
    p <- p + ggtitle(argv$title)
}

df.smooth <- df.in %>% filter(grepl("Ring.*some16", Algo))

if (nrow(df.smooth %>% filter(grepl(".*Testsome.*", Algo))) > 0) {
    df.smooth <- df.smooth %>% filter(Nodes %in% c(2,4,8,16,32))
}

p1 <- ggplot(data = df.in, aes(x = factor(Nodes), y = median)) +
    geom_line(
        aes(colour = Algo, group = Algo), position = pd) +
    scale_y_continuous(trans="log2") +
    scale_colour_brewer(type="qal", palette="Paired") +
    geom_point(
        aes(colour = Algo, group = Algo, shape=Algo),# & Nodes %in% c(2,4,8,16,32)
        position = pd, size = 2) +
    geom_smooth(data = df.smooth, aes(x = as.numeric(factor(Nodes)), y=median), method="lm", se = FALSE, linetype = "dashed", size=1, fullrange=TRUE) +
    theme_bw()  + xlab('Nodes') + ylab('# Test / Wait Calls')

cowplot::plot_grid(p, p1, ncol=1)

## Here we add our special class
#if(!exists("print.gg_multiple", mode="function")) {
#    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
#}
#class(p) <- c('gg_multiple', class(p))

#print(p)

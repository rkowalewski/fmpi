#!/usr/bin/env Rscript

suppressMessages(library(readr))
#suppressMessages(library(dplyr))
library(ggplot2)
library(RColorBrewer)
library(tikzDevice)
#library(extrafont)


if(!exists("summarySE", mode="function")) {
    # Let us source the summary.R script
    initial.options <- commandArgs(trailingOnly = FALSE)
    file.arg.name <- "--file="
    script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
    script.basename <- dirname(script.name)
    other.name <- file.path(script.basename, "./include/summary.R")
    source(other.name)
}

#args = commandArgs(trailingOnly=TRUE)
#
#if (length(args)< 2) {
#    stop("Usage: ./stats.R <infile>", call.=FALSE)
#}

#csv.data <- read.csv(f, header=TRUE, strip.white=TRUE)
f <- file("stdin")
df.in <- read_csv(paste(collapse = "\n", readLines(f)), col_names=TRUE, col_types="iiiiicid")
close(f)

df.summary <- summarySE(df.in,
                          measurevar="Time",
                          groupvars=c("Nodes", "Procs", "Round", "Algo"),
                          na.rm=TRUE)

df.minValues <- df.in %>%
    group_by(Nodes, Procs, Round, Algo) %>%
    slice(which.min(Time))

df.maxValues <- df.in %>%
    group_by(Nodes, Procs, Round, Algo) %>%
    slice(which.max(Time))

df.summary$minRank <- df.minValues$Rank
df.summary$maxRank <- df.maxValues$Rank

#df.summary$minRank <- df.in$Rank[[df.summary$minRow]]
cat(format_csv(df.summary))


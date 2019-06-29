#!/usr/bin/env Rscript

suppressMessages(library(readr))
suppressMessages(library(dplyr))
#library(ggplot2)
#library(RColorBrewer)
#library(tikzDevice)
#library(extrafont)


args = commandArgs(trailingOnly=TRUE)

if (length(args)< 2) {
    stop("Usage: ./stats.R <infile> <outPrefix>", call.=FALSE)
}

f <- args[1]
csv.data <- read.csv(f, header=TRUE, strip.white=TRUE)

outputCsvPrefixPath <- args[2]

winners <- csv.data %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, PPN) %>%
    top_n(-3, Tmedian) %>%
    arrange(PPN)

winners <- winners[c("Nodes", "Procs", "PPN", "NBytes", "Blocksize", "Algo", "Tmedian",  "speedup")]

write.csv(winners, paste(outputCsvPrefixPath, ".winners.csv", sep=""), row.names=FALSE)

count <- winners %>% group_by(Algo, PPN) %>% tally() %>% arrange(PPN, -n)

write.csv(count, paste(outputCsvPrefixPath, ".count.csv", sep=""), row.names=FALSE)

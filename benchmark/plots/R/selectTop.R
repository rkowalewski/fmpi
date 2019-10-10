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

# Obtain the best own algorithms witout baseline
best <- csv.data %>%
    filter(!(Algo == "AlltoAll")) %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, PPN) %>%
    top_n(-3, Ttotal_median) %>% ungroup()

a2a <- csv.data %>%
    filter(Algo == "AlltoAll")

# combine our own best algorthms and alltoall (baseline)
combined <- rbind(best, a2a)
# sort groupwise by median
combined <- combined %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, PPN) %>%
    arrange(PPN, Blocksize, Procs, Ttotal_median)

# reorder columns
combined <- combined %>%
    select(Nodes, Procs, PPN, NBytes, Blocksize, Cat, Algo,
           Ttotal_speedup, Ttotal_median, Ttotal_mean, everything())

# write it out
write.csv(combined, paste(outputCsvPrefixPath, ".top3.csv", sep=""), row.names=FALSE)

# Select the lonely winner in each measurement
winnersCat <- combined %>% slice(1)

countCat <- winnersCat %>% group_by(Algo, PPN, Cat) %>% tally() %>% arrange(PPN, Cat, -n)
countCat$Percent <- round(countCat$n / nrow(a2a), digits=2)

count <- countCat %>% group_by(Algo, PPN) %>%
    summarise_at(vars(n, Percent), list(~sum(.))) %>% arrange(PPN, -n)

write.csv(countCat, paste(outputCsvPrefixPath, ".winnersByCat.csv", sep=""), row.names=FALSE)
write.csv(count, paste(outputCsvPrefixPath, ".winners.csv", sep=""), row.names=FALSE)

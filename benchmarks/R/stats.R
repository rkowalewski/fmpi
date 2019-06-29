#!/usr/bin/env Rscript

suppressMessages(library(readr))
suppressMessages(library(dplyr))
#library(ggplot2)
#library(RColorBrewer)
#library(tikzDevice)
#library(extrafont)


#args = commandArgs(trailingOnly=TRUE)
#
#if (length(args)< 2) {
#    stop("Usage: ./stats.R <infile>", call.=FALSE)
#}

#csv.data <- read.csv(f, header=TRUE, strip.white=TRUE)
f <- file("stdin")
df.in <- read_csv(paste(collapse = "\n", readLines(f)), col_names=TRUE, col_types="iii?iciddd")
close(f)

df.stats <- df.in %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, Algo) %>%
    summarize(
              N = sum(!is.na(Ttotal)),
              Tmedian = median(Ttotal, na.rm = TRUE),
              Tmean = mean(Ttotal, na.rm = TRUE),
              Tmin = min(Ttotal, na.rm = TRUE),
              Tmax = max(Ttotal, na.rm = TRUE),
              minRank = Rank[which.min(Ttotal)],
              maxRank = Rank[which.max(Ttotal)],
              sd = sd(Ttotal, na.rm = TRUE),
              Tcomm = median(Tcomm, na.rm = TRUE),
              Tmerge = median(Tmerge, na.rm = TRUE)
    ) %>%
    # add the speedup
    mutate(speedup = Tmedian[Algo == "AlltoAll"] / Tmedian) %>%
    arrange(Tmedian, .by_group = TRUE)



# Standard Error
df.stats$se <- df.stats$sd / sqrt(df.stats$N)
# CI Interval
ciInterval <- .95
ciMult <- qt(ciInterval/2 + .5, df.stats$N-1)
# Sum
df.stats$ci <- df.stats$se * ciMult

df.stats$PPN <- df.stats$Procs / df.stats$Nodes

cat(format_csv(df.stats))


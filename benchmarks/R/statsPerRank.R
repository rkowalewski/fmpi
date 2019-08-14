#!/usr/bin/env Rscript

suppressMessages(library(readr))
suppressMessages(library(dplyr))

addSeCi <- function(df, ciInterval=.95) {
    # Standard Error
    df$se <- df$sd / sqrt(df$N)
    # CI Interval
    ciMult <- qt(ciInterval/2 + .5, df$N-1)
    df$ci <- df$se * ciMult
    df
}

args = commandArgs(trailingOnly=TRUE)

if (length(args)< 1) {
    stop("Usage: ./statsPerRank.R <infile>", call.=FALSE)
}

f <- args[1]
df.in <- read_csv(f, col_names=TRUE, col_types="iii?icid??")

              #Ttotal_med = median(Ttotal, na.rm = TRUE),
              #Ttotal_mean = mean(Ttotal, na.rm = TRUE),
              #Ttotal_min = min(Ttotal, na.rm = TRUE),
              #Ttotal_max = max(Ttotal, na.rm = TRUE),
              #Tcomm_med = median(Tcomm, na.rm = TRUE),
              #Tmerge_med = median(Tmerge, na.rm = TRUE),
              #sd = sd(Ttotal, na.rm = TRUE),
              #Tcomm_sd = sd(Tcomm, na.rm = TRUE)
#    )
df.stats <- df.in %>%
    group_by(Nodes, Procs, NBytes, Blocksize, Algo, Rank) %>%
    summarise_at(
        vars(Ttotal, Tcomm, Tmerge),
        funs(median, mean, min, max, sd, n(),
             se=sd(.)/sqrt(n())
        )
    )


#df.stats <- addSeCi(df.stats, .95)

df.stats$PPN <- df.stats$Procs / df.stats$Nodes

cat(format_csv(df.stats))


#!/usr/bin/env Rscript

suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(DescTools))

f <- file("stdin")
df.in <- read_csv(paste(collapse = "\n", readLines(f)), col_names=TRUE, col_types="iii?iciddd")
close(f)

ci <- function(x, prob = .95) {
  n <- sum(!is.na(x))
  sd_x <- sd(x, na.rm = TRUE)
  z_t <- qt(1 - (1 - prob) / 2, df = n - 1)
  z_t * sd_x / sqrt(n)
}

lower <- function(x, prob = 0.95) {
  mean(x, na.rm = TRUE) - ci(x, prob)
}

upper <- function(x, prob = 0.95) {
  mean(x, na.rm = TRUE) + ci(x, prob)
}

medianCI <- function(x, prob=.95) {
    MedianCI(x,
         conf.level = 0.95,
         na.rm = TRUE,
         method = "exact",
         R = 10000)
}


ci_prob <- .95

df.stats <- df.in %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, Algo) %>%
    summarise_at(vars(Ttotal, Tcomm, Tmerge),
                  list(
                     N = ~sum(!is.na(.)),
                     ~mean(., na.rm = TRUE),
                     ~median(., na.rm = TRUE),
                     ~min(., na.rm = TRUE),
                     minR = ~Rank[which.min(.)],
                     ~max(., na.rm = TRUE),
                     maxR = ~Rank[which.max(.)],
                     ~sd(., na.rm = TRUE),
                     se=~sd(., na.rm = TRUE)/sqrt(sum(!is.na(.))),
                     avg_ci=~ci(., ci_prob),
                     med_lowerCI=~medianCI(., ci_prob)[2],
                     med_upperCI=~medianCI(., ci_prob)[3]
                  )
                ) %>%
    # add the speedup
    mutate(
           Ttotal_speedup = Ttotal_median[Algo == "AlltoAll"] / Ttotal_median,
           Tcomm_speedup = Tcomm_median[Algo == "AlltoAll"] / Tcomm_median
          ) %>%
    # Sort by the fastest in each group
    arrange(Ttotal_median, .by_group = TRUE)


minThreshold <- 1024
mediumThreshold <- 16384
df.stats$Cat <- cut(df.stats$Blocksize, c(0,minThreshold,mediumThreshold,Inf), c("small", "medium", "large"))
df.stats$PPN <- df.stats$Procs / df.stats$Nodes

df.stats <- df.stats %>%
    select(Nodes, Procs, PPN, Round, NBytes, Blocksize, Cat, Algo,
           Ttotal_speedup, Ttotal_median,
           Ttotal_min, Ttotal_max,
           Ttotal_med_lowerCI,Ttotal_med_upperCI,
           everything())


cat(format_csv(df.stats))


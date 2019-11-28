#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))
suppressMessages(library(DescTools))

# Convert str input to boolean
str2bool = function(input_str)
{
  if(input_str == "0")
  {
    input_str = FALSE
  }
  else if(input_str == "1")
  {
    input_str = TRUE
  }
  return(input_str)
}

args = commandArgs(trailingOnly=TRUE)

if (length(args) < 2) {
    stop("Usage: ./plots.R <csv_input> <csv_output>", call.=FALSE)
}

csv_in <- args[1]
csv_out <- args[2]

print(paste0("--reading file: ", csv_in))

df.in <- read_csv(csv_in, col_names=TRUE, col_types="iiii?iciicd")

ci <- function(n, sd_x, prob = .95) {
  z_t <- qt(1 - (1 - prob) / 2, df = n - 1)
  z_t * sd_x / sqrt(n)
}
#
#lower <- function(x, prob = 0.95) {
#  mean(x, na.rm = TRUE) - ci(x, prob)
#}
#
#upper <- function(x, prob = 0.95) {
#  mean(x, na.rm = TRUE) + ci(x, prob)
#}
#
medianCI <- function(x, prob=.95) {
    MedianCI(x,
         conf.level = 0.95,
         na.rm = TRUE,
         method = "exact",
         R = 10000)
}


ci_prob <- .95

print("--calulating statistics")

if (!("Threads" %in% colnames(df.in))) {
    df.in <- df.in %>% mutate(Threads = 0)
}

df.stats <- df.in %>%
    group_by(Nodes, Procs, Threads, Blocksize, Algo, Measurement) %>%
    summarise_at(
              vars(c("Value")),
                  list(
                     n = ~sum(!is.na(.)),
                     ~mean(., na.rm = TRUE),
                     ~median(., na.rm = TRUE),
                     ~sd(., na.rm = TRUE),
                     se=~sd(., na.rm = TRUE)/sqrt(sum(!is.na(.))),
                     ~min(., na.rm = TRUE),
                     ~max(., na.rm = TRUE),
                     #,minR = ~Rank[which.min(.)],
                     #maxR = ~Rank[which.max(.)],
                     med_lowerCI=~medianCI(., ci_prob)[2],
                     med_upperCI=~medianCI(., ci_prob)[3]
                  )
                ) %>%
    ungroup() %>%
    mutate(avg_ci = ci(n, sd, ci_prob),
           PPN = Procs / Nodes)

print(paste0("--writing file: ", csv_out))

write_csv(df.stats, csv_out, na = "NA", append = FALSE, col_names = TRUE,
            quote_escape = "double")

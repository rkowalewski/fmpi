#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))
suppressMessages(library(DescTools))
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

# load util functions
source(paste0(dirname(thisFile()), "/internal/", "utils.R"))

params <- arg_parser("calculating statistics from benchmark output")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "-")

argv <- parse_args(params)

df.in <- util.read_csv(argv$input, col_names=TRUE, col_types="iii?icicd")

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
                     minR = ~Rank[which.min(.)],
                     maxR = ~Rank[which.max(.)],
                     med_lowerCI=~medianCI(., ci_prob)[2],
                     med_upperCI=~medianCI(., ci_prob)[3]
                  )
                ) %>%
    #arrange(median, .by_group = TRUE) #%>%
    ungroup() %>%
    # filter(Measurement == "Ttotal") %>%
    mutate(avg_ci = ci(n, sd, ci_prob),
            PPN = Procs / Nodes)

util.write_csv(df.stats, argv$output)

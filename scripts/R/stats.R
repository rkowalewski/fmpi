#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))

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

if (length(args) < 3) {
    stop("Usage: ./plots.R <csv_input> <csv_output> <append?>", call.=FALSE)
}

csv_in <- args[1]
csv_out <- args[2]

append <- str2bool(args[3])
append <- FALSE

print(paste0("--reading file: ", csv_in))

df.in <- read_csv(csv_in, col_names=TRUE, col_types="iii?iciicd")

#ci <- function(x, prob = .95) {
#  n <- sum(!is.na(x))
#  sd_x <- sd(x, na.rm = TRUE)
#  z_t <- qt(1 - (1 - prob) / 2, df = n - 1)
#  z_t * sd_x / sqrt(n)
#}
#
#lower <- function(x, prob = 0.95) {
#  mean(x, na.rm = TRUE) - ci(x, prob)
#}
#
#upper <- function(x, prob = 0.95) {
#  mean(x, na.rm = TRUE) + ci(x, prob)
#}
#
#medianCI <- function(x, prob=.95) {
#    MedianCI(x,
#         conf.level = 0.95,
#         na.rm = TRUE,
#         method = "exact",
#         R = 10000)
#}


ci_prob <- .95

# commonVars <- unlist(df.in %>% distinct(Measurement))
commonVars <- c("Ttotal", "Tcomm", "Tcomp")

print("--calulating common statistics")

df.stats <- df.in %>%
    filter(Measurement %in% commonVars) %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, Algo, Measurement) %>%
    spread(Measurement, Value) %>%
    summarise_at(vars(commonVars),
                  list(
                     N = ~sum(!is.na(.)),
                     ~mean(., na.rm = TRUE),
                     ~median(., na.rm = TRUE),
                     ~min(., na.rm = TRUE),
                     ~max(., na.rm = TRUE),
                     ~sd(., na.rm = TRUE),
                     se=~sd(., na.rm = TRUE)/sqrt(sum(!is.na(.)))
                     #,minR = ~Rank[which.min(.)],
                     #maxR = ~Rank[which.max(.)],
                     #avg_ci=~ci(., ci_prob),
                     #med_lowerCI=~medianCI(., ci_prob)[2],
                     #med_upperCI=~medianCI(., ci_prob)[3]
                  )
                ) %>%

    # add speedup and other attributes
    mutate(
           Ttotal_speedup = Ttotal_median[Algo == "AlltoAll"] / Ttotal_median,
           Tcomm_speedup = Tcomm_median[Algo == "AlltoAll"] / Tcomm_median,
           PPN = Procs / Nodes
          ) %>%
    ## Sort by the fastest in each group
    arrange(Ttotal_median, .by_group = TRUE)

#df.stats$PPN <- df.stats$Procs / df.stats$Nodes

print("--calulating bruck specific statistics")

df.stats <- df.stats %>%
    select(Nodes, Procs, PPN, Round, NBytes, Blocksize,# Cat,
           Algo,
           Ttotal_speedup, Ttotal_median,
           Tcomm_median, Tcomp_median,
           Ttotal_min, Ttotal_max,
           #Ttotal_med_lowerCI,Ttotal_med_upperCI,
           everything())


df.bruck <- df.in %>%
    filter(grepl("^Bruck", Algo)) %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, Algo, Measurement) %>%
    summarise(median=median(Value, na.rm = T)) %>%
    mutate(
           Percent = median/median[Measurement=="Ttotal"]
          )

#minThreshold <- 1024
#mediumThreshold <- 16384
#df.stats$Cat <- cut(df.stats$Blocksize, c(0,minThreshold,mediumThreshold,Inf), c("small", "medium", "large"))
#


{ if (append == TRUE){
    print(paste0("--appending file: ", csv_out))
  }
  else {
    print(paste0("--writing file: ", csv_out))
  }
}



write_csv(df.stats, csv_out, na = "NA", append = FALSE, col_names = TRUE,
            quote_escape = "double")

csv_bruck <- paste(
                  sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv_out)),
                  ".bruck.csv", sep="")

print(paste0("--writing file: ", csv_bruck))

write_csv(df.bruck, csv_bruck, na = "NA", append = FALSE, col_names = TRUE,
            quote_escape = "double")

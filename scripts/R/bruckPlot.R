#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))
suppressMessages(library(ggforce))
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

params <- arg_parser("visualizing times in a stacked bar plot")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")

argv <- parse_args(params)

df.in <- 0

if (argv$input == "-") {
    df.in <- read_csv(file('stdin'), col_names=TRUE, col_types="iiiiccidddddddddi")
} else {
    df.in <- read_csv(file=argv$input, col_names=TRUE, col_types="iiiiccidddddddddi")
}


levels <- c("Tcomm", "Tcomp", "Trotate", "Tpack", "Tunpack", "Ttotal")

df.in <- df.in %>%
    filter(Measurement %in% levels) %>%
    mutate(Measurement=parse_factor(Measurement, levels)) %>%
    ungroup()

if (!("PPN" %in% colnames(df.in))) {
    df.in <- df.in %>% mutate(PPN = Procs / Nodes)
}

df.bar <- df.in %>% filter(Measurement != "Ttotal" )
df.pnt <- df.in %>% filter(Measurement == "Ttotal" )
df.a2a <- df.in %>% filter(Measurement == "Ttotal" & Algo == "AlltoAll")

pdf(argv$output,paper="a4")

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 45)

my_labeller <- label_bquote(
  cols = .(Nodes)
)

pd <- position_dodge(0.1)

p <- ggplot(data=df.pnt, aes(x=Algo, y=median, group=Algo)) +
    # Bar Plot
    geom_bar(data=df.bar, aes(y = median, x = Algo, fill = Measurement), stat="identity",
        position='stack', alpha=0.8) +
    # Errorbars for confidence interval
    geom_errorbar(
        aes(ymin=med_lowerCI, ymax=med_upperCI), colour="black", width=.1, position=pd) +
    # Plotting the median of all algorithms
    geom_point(position=pd, size=2, shape=23) +
    #geom_hline(data=df.a2a, aes(yintercept=median), colour="black", linetype="dashed") +
    # To use for line and point colors, add
    scale_fill_brewer(type="qal", palette="Paired") +
    theme +
    facet_wrap_paginate(Threads~PPN~Blocksize~Nodes, ncol=3, nrow=3,
        scales="free_y", page=NULL, labeller = my_labeller)

# other labeller are:
# label_wrap_gen(multi_line=FALSE)
#

# Here we add our special class
if(!exists("print.gg_multiple", mode="function")) {
    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
}
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


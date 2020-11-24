#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(gdata))
suppressMessages(library(argparser))
suppressMessages(library(ggsci))
suppressMessages(library(tikzDevice))

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
source(paste0(dirname(thisFile()), "/../R/internal/", "utils.R"))

params <- arg_parser("visualizing speedup in a lineplot")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "")
params <- add_argument(params, "--title", help = "PDF Title", default = "")
params <- add_argument(params, "--caption", help = "PDF Caption", default = "")
params <- add_argument(params, "--device", help = "Output device (pdf, tikz)", default = "pdf")
params <- add_argument(params, "--paper", help = "paper format (see R pdf manual)", default = "special")

argv <- parse_args(params)

isPow2 <- function(x) {
    ifelse(x < 2,FALSE,!any(as.logical(intToBits(x) & intToBits(x-1))))
}


# just declare this variable
data <- 0

if (argv$input == "-") {
    data <- read.csv(file=file('stdin'), header=TRUE, sep=",")
} else {
    data <- read.csv(file=argv$input, header=TRUE, sep=",")
}

if (nrow(data) == 0) {
    stop("no data available")
}

best <- data %>%
    filter(bench != "Baseline") %>%
    group_by(size) %>%
    top_n(-1, total) %>%
    ungroup() %>%
    mutate(bench="FMPI")

baseline <- data %>% filter(bench == "Baseline")

# combine our own best algorthms and baseline
combined <- rbind(best, baseline) %>%
    mutate(
         hr=humanReadable(size, digits=0, standard="Unix"),
         winsz=ifelse(bench == "Baseline" | bench == "Bruck", 1, winsz),
         r_comp = 100 * compute / total,
         r_wait = 100 * mpi.wait / total,
         r_init = 100 * init / total,
         )

c_long <- gather(combined, condition, measurement, overlap,r_comp,r_wait,r_init, factor_key=TRUE) %>% arrange(size)

cat(format_csv(c_long))

title <- argv$title

if (argv$output == '') {
    argv$output = paste0(remove_extension(basename(argv$input)),".ialltoall.overlap")
}

if (title == '') {
    title <- remove_extension(basename(argv$output))
}



if (argv$device == 'pdf') {
    pdf(paste0(argv$output, ".pdf"), title=title, paper=argv$paper)
} else {
    tikz(file=paste0(argv$output, ".tex"), width=5, height=5)
}


theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 90)
# theme$axis.title.x=element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))
# theme$legend.justification = c(0.9, 0)
theme$legend.position = "top"
theme$legend.title = element_blank()
# theme$legend.background = element_rect(size=.2)
#theme$legend.position=c(0.02,0.98)
#theme$legend.justification=c(0.02,0.98)
#theme$legend.box="horizontal";
# theme$legend.direction="horizontal";

xvar <- "hr"
xlab <- "Message Size (bytes)"

yvar <- "measurement"

ylab <- "Total Time (%)"

if (argv$device == 'tikz') {
    ylab <- "Total Time (\\%)"
}


fill <- "condition"

# ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']

# This is even colorblind safe
# ['#1b9e77','#d95f02','#7570b3']

# Ring and Hypercube are never used at the same time
# so we assign the same color

# We want to filter window size == 1 because it is slow anyay
# data <- data %>% filter((!(bench %in% algos_window)) | winsz > 1)
# so we also have to set the legend values
# winsizes <- data %>% filter(winsz > 1) %>% distinct(winsz) %>% pull(winsz)

# The errorbars overlapped, so use position_dodge to move them horizontally
# pd <- position_dodge(0.1) # move them .05 to the left and right

p <- ggplot(data = c_long %>% filter(condition != "overlap"), aes_string(x=paste0("factor(", xvar, ")"), y=yvar
                             )) +
    geom_bar(aes(fill=condition), position="stack", width = 0.8, stat="identity") +
    scale_x_discrete(labels = c_long %>% distinct(hr) %>% pull(hr)) +
    scale_fill_manual(values=c("#1b9e77","#d95f02","#7570b3"),
                      name="Type",
                      breaks=c("r_init", "r_comp", "r_wait"),
                      labels=c("Schedule", "Computation", "Wait")
                    ) +
    geom_point(data = c_long %>% filter(condition == "overlap"), shape=4) +
    facet_wrap(~bench, nrow = 1) + #+ theme(panel.spacing = unit(0, "lines"))
    theme +
    ylab(ylab) + xlab(xlab)


print(p)

# suppress null device output
garbage <- dev.off()


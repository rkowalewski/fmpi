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
params <- add_argument(params, "--speedup", help = "Output device (pdf, tikz)", flag = TRUE)

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

data <- data %>%
    # filter(size <= 128 * 1024) %>%
    group_by(nodes,procs,size) %>%
    mutate(speedup = total[bench == "Baseline"] / total) %>%
    ungroup() %>%
    mutate(
         hr=humanReadable(size, digits=0, standard="Unix"),
         winsz=ifelse(bench == "Baseline" | bench == "Bruck", 1, winsz),
         bench=ifelse(bench == "Ring" & isPow2(nodes * procs), "Hypercube", bench)
    )

title <- argv$title

if (argv$output == '') {
    argv$output = paste0(remove_extension(basename(argv$input)),".ialltoall")
    if (argv$speedup) {
        argv$output = paste0(argv$output,".speedup")
    }
}

if (title == '') {
    title <- remove_extension(basename(argv$output))
}



if (argv$device == 'pdf') {
    pdf(paste0(argv$output, ".pdf"), title=title, paper=argv$paper)
} else {
    tikz(file=paste0(argv$output, ".tex"), width=6, height=4)
}


theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 90)
theme$axis.title.x=element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))
# theme$legend.justification = c(0.9, 0)
# theme$legend.position = "top"
# theme$legend.title = element_blank()
# theme$legend.background = element_rect(size=.2)
theme$legend.position=c(0.02,0.98)
theme$legend.justification=c(0.02,0.98)
theme$legend.box="horizontal";
# theme$legend.direction="horizontal";

xvar <- "size"
xlab <- "Message Size (bytes)"

yvar <- "total"
ylab <- "Total Time (usecs)"

color <- "bench"
shape <- "winsz"

if (argv$speedup) {
    ylab <- "Speedup"
    yvar <- "speedup"
    data <- data %>% filter(bench != "Baseline")
}

# '#b2182b','#ef8a62','#fddbc7','#d1e5f0','#67a9cf','#2166ac'
# ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']
# ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']

# Ring and Hypercube are never used at the same time
# so we assign the same color
colors <- c("OneFactor" = "#e41a1c",
            "Ring" = "#377eb8",
            "Hypercube" = "#377eb8",
            "Bruck" = "#4daf4a",
            "Baseline" = "#000000",
            "FMPI" = "#984ea3")

linetypes <- c("OneFactor" = "dashed",
            "Ring" = "dotted",
            "Hypercube" = "dotted",
            "Bruck" = "twodash",
            "Baseline" = "solid",
            "FMPI" = "solid")

algos_window <- c("OneFactor", "Ring", "Hypercube")

# We want to filter window size == 1 because it is slow anyay
data <- data %>% filter((!(bench %in% algos_window)) | winsz > 1)
# so we also have to set the legend values
winsizes <- data %>% filter(winsz > 1) %>% distinct(winsz) %>% pull(winsz)

xlabels <- data %>% distinct(hr) %>% pull()

# The errorbars overlapped, so use position_dodge to move them horizontally
# pd <- position_dodge(0.1) # move them .05 to the left and right

p <- ggplot(data, aes_string(x=paste0("factor(", xvar, ")"), y=yvar,
                             group="interaction(bench,winsz)", shape=paste0("factor(", shape, ")"), colour=color, linetype=color))

p <- p +
    geom_line() +
    theme +
    scale_x_discrete(labels=xlabels) +
    scale_linetype_manual(name="Algorithm", values=linetypes) +
    scale_color_manual(values=colors, name="Algorithm") +
    scale_shape_discrete(name="Window Size", na.translate=FALSE, breaks=winsizes) +
    guides(color = guide_legend(override.aes = list(shape = NA),
                                title.position="top"), # add title.hjust = 0.5 if title should be more centered
            shape = guide_legend(title.position="top")
           ) +
    ylab(ylab) + xlab(xlab)

if (n_distinct(data$bench) > 1) {
 p <- p + geom_point(data = data %>% filter(bench %in% algos_window), size=1.5)
}


if (isFALSE(argv$speedup)) {
    p <- p + scale_y_continuous(trans='log10')
}

if (argv$speedup) {
    mylimit <- function(x) {
        limits <- c(min(x) - .1, max(x) + .1)
        if (n_distinct(data$bench) == 1) {
            limits[1] <- 1.0
        }
        limits
    }

    p <- p + scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit)

    if (n_distinct(data$bench) > 1) {
        p <- p + geom_hline(yintercept = 1, color = unname(colors["Baseline"]))
    }
}

if (argv$caption != '') {
   # p <- p + ggtitle(argv$caption)
}

print(p)

# suppress null device output
garbage <- dev.off()


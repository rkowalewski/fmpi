#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(gdata))
#suppressMessages(library(gridExtra))
#suppressMessages(library(grid))
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

# just declare this variable
data <- 0

if (argv$input == "-") {
    data <- read.csv(file=file('stdin'), header=TRUE, sep=",")
} else {
    data <- read.csv(file=argv$input, header=TRUE, sep=",")
}

data <- data %>%
    filter(size <= 128 * 1024) %>%
    group_by(nodes,procs,size) %>%
    mutate(speedup = total[bench == "Baseline"] / total) %>%
    ungroup()

data <- data %>%
    mutate(
         hr=humanReadable(size, digits=0, standard="Unix" ),
           algo = ifelse(bench == "Baseline", bench,
                  sub(pattern = "(.*)\\.(.*)\\.(.*)$", replacement = "\\1", bench)),
           winsz = ifelse(bench == "Baseline", NA,
                  sub(pattern = "(.*)\\.(.*)\\.(.*)$", replacement = "\\3", bench))
    )

if (nrow(data) == 0) {
    stop("no data available")
}


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
    tikz(file=paste0(argv$output, ".tex"), width=5, height=5)
}

x_variable <- "size"

ylab <- "Total Time (usecs)"
yvar <- "total"
color <- "algo"
shape <- "winsz"

if (argv$speedup) {
    ylab <- "Speedup"
    yvar <- "speedup"
    data <- data %>% filter(bench != "Baseline")
}

# cat(format_csv(data[data$bench != "Baseline",]))


theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 90)
# theme$legend.justification = c(0.9, 0)
theme$legend.position = "top"
# theme$legend.title = element_blank()
theme$legend.background = element_rect(size=.2)
theme$axis.title.x=element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))

# data <- data %>% filter(algo != "Ring")

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right

p <- ggplot(data, aes_string(x=paste0("factor(", x_variable, ")"), y=yvar, group="bench"))
p <- p + geom_line(aes_string(linetype=color, colour=color)) +
    geom_point(data = data %>% filter(bench != "Baseline"), size=2,
               aes_string(shape=shape,colour=color)) +
    theme +
    scale_x_discrete(labels=data$hr) +
    # To use for line and point colors, add
    # scale_colour_grey() +
    # scale_colour_brewer() +
    scale_color_npg() +
    guides(color = guide_legend(override.aes = list(shape = NA))) +
    ylab(ylab) + xlab("Message Size (bytes)")

# if (isFALSE(argv$speedup)) {
#     p <- p + labs(shape="Window Size", colour="Algorithm", linetype="Algorithm")
# }

if (argv$speedup) {
    mylimit <- function(x) {
        limits <- c(min(x) - .2, max(x) + .2)
        limits
    }

    p <- p + geom_hline(yintercept = 1) +
    scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit)
}

if (argv$caption != '') {
    p <- p + ggtitle(argv$caption)
}

print(p)

# suppress null device output
garbage <- dev.off()


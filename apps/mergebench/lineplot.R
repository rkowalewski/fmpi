#!/usr/bin/env Rscript

# suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
# suppressMessages(library(gridExtra))
# suppressMessages(library(grid))
suppressMessages(library(argparser))
suppressMessages(library(ggsci))



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
source(paste0(dirname(thisFile()), "/../../scripts/R/internal/", "utils.R"))


params <- arg_parser("visualizing mergebench performance")
params <- add_argument(params, "--input", default = "-", help = "input file")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")
params <- add_argument(params, "--title", help = "PDF Title", default = "")
params <- add_argument(params, "--caption", help = "PDF Caption", default = "")
params <- add_argument(params, "--paper", help = "paper format (see R pdf manual)",
    default = "special")


argv <- parse_args(params)

if (argv$input == "-") {
    data <- read.csv(file = file("stdin"), header = TRUE, sep = ",")
} else {
    data <- read.csv(file = argv$input, header = TRUE, sep = ",")
}

mylimit <- function(x) {
    limits <- c(min(x) - .2, max(x) + .2)
    limits
}

title <- argv$title

if (title == '') {
    title <- remove_extension(basename(argv$output))
}

pdf(argv$output, title=title, paper=argv$paper)

x_variable <- "Blocksize"
y_variable <- "Time"
group_by <- "Algorithm"
y_lab <- "Time (sec)"

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 45)

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right
p <- ggplot(data, aes_string(x=paste0("factor(", x_variable, ")"), y=y_variable, group=group_by))
p <- p + geom_line(position=pd, aes(linetype=Algorithm, colour=Algorithm)) +
    geom_point(position=pd, size=2,aes(shape=Algorithm, colour=Algorithm)) +
    theme +
    # To use for line and point colors, add
    # scale_colour_grey() +
    # scale_colour_brewer() +
    scale_color_npg() +
    #scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
    xlab(x_variable) +
    ylab(y_lab)
    #+geom_hline(yintercept = 1)

if (argv$caption != '') {
    p <- p + ggtitle(argv$caption)
}

print(p)

# suppress null device output
garbage <- dev.off()


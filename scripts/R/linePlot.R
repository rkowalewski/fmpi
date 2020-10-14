#!/usr/bin/env Rscript

# suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
#suppressMessages(library(gridExtra))
#suppressMessages(library(grid))
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
source(paste0(dirname(thisFile()), "/internal/", "utils.R"))

params <- arg_parser("visualizing speedup in a lineplot")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")
params <- add_argument(params, "--title", help = "PDF Title", default = "")
params <- add_argument(params, "--caption", help = "PDF Caption", default = "")
params <- add_argument(params, "--paper", help = "paper format (see R pdf manual)", default = "special")

argv <- parse_args(params)

# just declare this variable
data.in <- 0


if (argv$input == "-") {
    data.in <- read.csv(file=file('stdin'), header=TRUE, sep=",")
} else {
    data.in <- read.csv(file=argv$input, header=TRUE, sep=",")
}

data <- data.in %>%
    filter(Measurement == "Ttotal") %>%
    group_by(Nodes,Procs,Threads, Blocksize) %>%
    mutate(Ttotal_speedup = median[Algo == "Alltoall"] / median) %>%
    ungroup() %>%
    filter(Algo != "Alltoall")

if (nrow(data) == 0) {
    stop("no data available")
}

if (!("Ttotal_speedup" %in% colnames(data))) {
    stop("something went wrong. We cannot calculate the speedup in this dataset.")
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

my_labeller <- label_bquote(
  cols = .(PPN) / .(Threads) / .(Blocksize)
)

x_variable <- "Blocksize"

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 45)

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right
p <- ggplot(data, aes_string(x=paste0("factor(", x_variable, ")"), y="Ttotal_speedup", group="Algo"))
p <- p + geom_line(position=pd, aes(linetype=Algo, colour=Algo)) +
    geom_point(position=pd, size=2,aes(shape=Algo, colour=Algo)) +
    theme +
    # To use for line and point colors, add
    # scale_colour_grey() +
    # scale_colour_brewer() +
    scale_color_npg() +
    scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
    xlab(x_variable) +
    ylab("Speedup")+
    geom_hline(yintercept = 1)

if (argv$caption != '') {
    p <- p + ggtitle(argv$caption)
}

print(p)

# suppress null device output
garbage <- dev.off()


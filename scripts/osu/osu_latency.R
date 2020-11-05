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
params <- add_argument(params, "--paper", help = "paper format (see R pdf manual)", default = "special")
params <- add_argument(params, "--device", help = "Output device (pdf, tikz)", default = "pdf")

argv <- parse_args(params)

# just declare this variable
data <- 0

if (argv$input == "-") {
    data <- read.csv(file=file('stdin'), header=TRUE, sep=",")
} else {
    data <- read.csv(file=argv$input, header=TRUE, sep=",")
}

sizeMax <- 32 * 1024

data <- data %>%
     filter(size <= sizeMax) %>%
     mutate(hr=humanReadable(size, digits=0, standard="Unix" ))

head(data)

if (nrow(data) == 0) {
    stop("no data available")
}


title <- argv$title

if (argv$output == '') {
    argv$output = paste0(remove_extension(basename(argv$input)),".latency")
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

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 90)
# theme$legend.justification = c(0.9, 0)
theme$legend.position = "top"
theme$legend.title = element_blank()
theme$axis.title.x=element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right
p <- ggplot(data, aes_string(x=paste0("factor(", x_variable, ")"), y="latency", group="bench"))
p <- p + geom_line(position=pd, aes(linetype=bench, colour=bench)) +
    geom_point(position=pd, size=2,aes(shape=bench, colour=bench)) +
    theme +
    scale_x_discrete(labels=data$hr) +
    # To use for line and point colors, add
    # scale_colour_grey() +
    # scale_colour_brewer() +
    scale_color_npg() +
    ylab("Latency (usecs)") + xlab("Message Size (bytes)")
    # facet_zoom(x = size < 200)

if (argv$caption != '') {
    p <- p + ggtitle(argv$caption)
}

print(p)

# suppress null device output
garbage <- dev.off()


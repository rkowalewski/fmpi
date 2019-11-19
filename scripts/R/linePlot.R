#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(gridExtra))
library(grid)

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

args = commandArgs(trailingOnly=TRUE)

if (length(args)< 1) {
    stop("Usage: ./plots.R <infile>", call.=FALSE)
}

csv <- args[1]

data.in <- read.csv(file=csv, header=TRUE, sep=",") %>%
    filter(Measurement == 'Ttotal')

data <- data.in %>%
    group_by(Nodes,Procs,Blocksize) %>%
    mutate(Ttotal_speedup = median[Algo == "AlltoAll"] / median) %>%
    ungroup() %>%
    filter(Algo != "AlltoAll")

file <- sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv))

plotName <- paste0(file, ".pdf")

#cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right

mylimit <- function(x) {
    limits <- c(min(x) - .2, max(x) + .2)
    limits
}

if (!("PPN" %in% colnames(data))) {
    data <- data %>% mutate(PPN = Procs / Nodes)
}

nSizes <- data %>% distinct(Blocksize)
nPPN <- data %>% distinct(PPN)

plots <- list()

top <- 10

for(i in seq(1, nrow(nPPN))) {
    ppn <- unlist(nPPN[i,1])
    for(ii in seq(1, nrow(nSizes))) {
        bsize <- unlist(nSizes[ii,1])

        mydata <- data %>% filter(PPN == ppn & Blocksize == bsize)

        top5Algos <- mydata %>%
            group_by(Nodes) %>% top_n(top, Ttotal_speedup) %>% ungroup() %>%
            group_by(Algo) %>% tally() %>% top_n(top, n) %>% select(Algo)

        top5Bsize <- mydata %>% filter(Algo %in% unlist(top5Algos))

        p <- ggplot(top5Bsize, aes(x=factor(Nodes), y=Ttotal_speedup, colour=Algo, group=Algo))
        plotTitle <- paste0("PPN: ", ppn, " / ", "Blocksize: ", bsize, " Bytes")

        p <- p + geom_line(position=pd) +
            geom_point(position=pd, size=2) +
            labs(caption=plotTitle) +
            theme_bw() +
            # To use for line and point colors, add
            scale_colour_brewer(type="qal", palette="Paired") +
            scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
            xlab("Nodes") +
            ylab("Speedup")+
            geom_hline(yintercept = 1)
            #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
            # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
        idx <- (i-1) * nrow(nSizes) + ii
        plots[[idx]] <- p
    }
}

ml <- marrangeGrob(plots, nrow=2, ncol=1)
ggsave(file = plotName, ml, device="pdf", paper="a4")

##p <- ggplot()
#
#plotTitle <- ""
#
#NN <- nrow(data %>% distinct(Nodes))
#
#if (NN > 1) {
#    p <- ggplot(data, aes(x=factor(Nodes), y=Ttotal_speedup, colour=Algo, group=Algo))
#    plotTitle <- paste0(file, ": (Processors per Node / Blocksize in Bytes)")
#} else {
#    p <- ggplot(data, aes(x=factor(Blocksize), y=Ttotal_speedup, colour=Algo, group=Algo))
#    plotTitle <- paste0(file, ": (Processors per Node)")
#}
#
#
#my_labeller <- label_bquote(
#  cols = .(PPN) / .(Blocksize)
#)
#
#if (NN > 1) {
#    p <- p + facet_wrap_paginate(PPN~Blocksize, ncol=1, nrow=3, scales="free_y", page=NULL, labeller = my_labeller)
#} else {
#    p <- p + facet_wrap_paginate(~PPN, ncol=1, nrow=3, scales="free_y", page=NULL, labeller = "label_value")
#}
#
#class(p) <- c('gg_multiple', class(p))
#
## Here we add our special class
#if(!exists("print.gg_multiple", mode="function")) {
#    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
#}
#
#pdf(plotName,paper="a4", title=plotTitle)
##print(p, page=2)
#print(p)
#dev.off()
#
#warnings()


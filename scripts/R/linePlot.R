#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))


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

data <- read.csv(file=csv, header=TRUE, sep=",")

data <- data %>% filter(Algo != "AlltoAll")

plotTitle <- sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv))

plotName <- paste0(plotTitle, ".pdf")

#cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right

# bars won't be dodged!
pdf(plotName,paper="a4", title=plotTitle)

mylimit <- function(x) {
    limits <- c(min(x) - .2, max(x) + .2)
    limits
}


p <- ggplot(data, aes(x=factor(Nodes), y=Ttotal_speedup, colour=Algo, group=Algo)) +
    #geom_errorbar(aes(ymin=Ttotal_med_lowerCI, ymax=Ttotal_med_upperCI), colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=2) +
    ggtitle(plotTitle) +
    theme_bw() +
    # To use for line and point colors, add
    scale_colour_brewer(type="qal", palette="Paired") +
    scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
    xlab("Nodes") +
    ylab("Speedup")+
    geom_hline(yintercept = 1) +
    #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
    # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
    facet_wrap_paginate( ~Blocksize, ncol=1, nrow=3, scales="free_y", page=NULL, labeller="label_both")


# Here we add our special class
if(!exists("print.gg_multiple", mode="function")) {
    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
}
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


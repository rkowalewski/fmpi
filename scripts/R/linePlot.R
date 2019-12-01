#!/usr/bin/env Rscript

suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))
suppressMessages(library(gridExtra))
suppressMessages(library(grid))
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

topAlgosByGroup <- function(df, ntop) {
    df %>%
        # we first get the best algorithms over all node counts for current blocksize
        group_by(Nodes) %>% top_n(ntop, Ttotal_speedup) %>% ungroup() %>%
        # and then we sum by algorithms to get the top10
        group_by(Algo) %>% tally() %>% top_n(ntop, n) %>% select(Algo)
}


params <- arg_parser("visualizing speedup in a lineplot")
params <- add_argument(params, "--input", help = "input file", default = "-")
params <- add_argument(params, "--output", help = "output file", default = "output.pdf")

argv <- parse_args(params)

data.in <- 0

if (argv$input == "-") {
    data.in <- read.csv(file=file('stdin'), header=TRUE, sep=",")
} else {
    data.in <- read.csv(file=argv$input, header=TRUE, sep=",")
}


data <- data.in %>%
    filter(Measurement == "Ttotal") %>%
    group_by(Nodes,Procs,Threads, Blocksize) %>%
    mutate(Ttotal_speedup = median[Algo == "AlltoAll"] / median) %>%
    ungroup() %>%
    filter(Algo != "AlltoAll")
#cat(format_csv(data))

if (!("Ttotal_speedup" %in% colnames(data))) {
    stop("something went wrong. We cannot calculate the speedup in this dataset.")
}

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

nPPN <- data %>% distinct(PPN)

plots <- list()

ntop <- 5

plotIdx <- 1

plotTitle <- sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(argv$output))

pdf(argv$output,paper="a4", title=plotTitle)

my_labeller <- label_bquote(
  cols = .(PPN) / .(Threads) / .(Blocksize)
)

p <- ggplot(data, aes(x=factor(Nodes), y=Ttotal_speedup, colour=Algo, group=Algo))
p <- p + geom_line(position=pd) +
    geom_point(position=pd, size=2) +
    #labs(caption=plotTitle) +
    theme_bw() +
    # To use for line and point colors, add
    scale_colour_brewer(type="qal", palette="Paired") +
    scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
    xlab("Nodes") +
    ylab("Speedup")+
    geom_hline(yintercept = 1) +
    facet_wrap_paginate(PPN~Threads~Blocksize, ncol=1, nrow=3,
        scales="free_y", page=NULL, labeller = my_labeller)
    #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
    # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)

# Here we add our special class
if(!exists("print.gg_multiple", mode="function")) {
    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
}

class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


#for(i in seq(1, nrow(nPPN))) {
#    ppn <- unlist(nPPN[i,1])
#
#    ppnData <- data %>% filter(PPN == ppn)
#    nn <- nrow(ppnData %>% distinct(Nodes))
#    #print(paste0("nn: ", nn))
#
#    if ((nn > 1)) {
#        nSizes <- ppnData %>% distinct(Blocksize)
#        # group by blocksize
#        for(ii in seq(1, nrow(nSizes))) {
#            bsize <- unlist(nSizes[ii,1])
#
#            mydata <- ppnData %>% filter(Blocksize == bsize)
#            cat(format_csv(mydata))
#            #print(paste(ppn, bsize, nrow(mydata), sep=", "))
#            #stop()
#
#            top5Algos <- topAlgosByGroup(mydata, ntop)
#
#            top5Bsize <- mydata %>% filter(Algo %in% unlist(top5Algos))
#
#            p <- ggplot(top5Bsize, aes(x=factor(Nodes), y=Ttotal_speedup, colour=Algo, group=Algo))
#            plotTitle <- paste0("PPN: ", ppn, " / ", "Blocksize: ", bsize, " Bytes")
#
#            p <- p + geom_line(position=pd) +
#                geom_point(position=pd, size=2) +
#                labs(caption=plotTitle) +
#                theme_bw() +
#                # To use for line and point colors, add
#                scale_colour_brewer(type="qal", palette="Paired") +
#                scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
#                xlab("Nodes") +
#                ylab("Speedup")+
#                geom_hline(yintercept = 1)
#                #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
#                # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
#            plots[[plotIdx]] <- p
#            plotIdx <- plotIdx + 1
#        }
#    } else {
#        mydata <- ppnData
#
#        top5Algos <- topAlgosByGroup(mydata, ntop)
#
#        top5Bsize <- mydata %>% filter(Algo %in% unlist(top5Algos))
#
#        p <- ggplot(top5Bsize, aes(x=factor(Blocksize), y=Ttotal_speedup, colour=Algo, group=Algo))
#        plotTitle <- paste0("PPN: ", ppn, " / ", "Nodes: ", mydata[1,1])
#
#        p <- p + geom_line(position=pd) +
#            geom_point(position=pd, size=2) +
#            labs(caption=plotTitle) +
#            theme_bw() +
#            # To use for line and point colors, add
#            scale_colour_brewer(type="qal", palette="Paired") +
#            scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
#            xlab("Blocksize") +
#            ylab("Speedup")+
#            geom_hline(yintercept = 1)
#            #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
#            # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
#        plots[[plotIdx]] <- p
#        plotIdx <- plotIdx + 1
#    }
#}
#
#ml <- marrangeGrob(plots, nrow=2, ncol=1)
#ggsave(file = argv$output, ml, device="pdf", paper="a4")

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


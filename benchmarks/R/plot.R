#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(tikzDevice)
library(extrafont)



plotLineChart <- function(data, pxlab, group_by, plabels) {
    # The errorbars overlapped, so use position_dodge to move them horizontally
    pd <- position_dodge(0.1) # move them .05 to the left and right

    ggplot(data, aes(x=factor(nnodes), y=Tmedian, colour=Algo, group=Algo)) +
    geom_errorbar(aes(ymin=time-se, ymax=time+se), colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=1, shape=21, fill="white") + # 21 is filled circle
    xlab(pxlab) +
    ylab("Median Time (s)") +
    # scale_colour_brewer(palette="Paired", name="Tasks / Threads") +     # Legend label, use darker colors
    scale_colour_brewer(palette="Paired", name="Algorithm"     # Legend label, use darker colors
                     #,breaks=Algo
                     ) +
                     #l=40) +                    # Use darker colors, lightness=40
    #ggtitle("The Effect of Vitamin C on\nTooth Growth in Guinea Pigs") +
    expand_limits(y=0) +                        # Expand y range
    #scale_y_continuous(breaks=0:50:5) +         # Set tick every 4
    theme_bw() +
    # we can plot the legend into the plot,
    # or outside the plot (comment it out)
    theme(legend.justification=c(1, 0),
        # Position legend in top right
          legend.position=c(1,0))
          #text=element_text(family="Linux Biolinum"))
    # use fc-lists to see a list of available fonts...

}

args = commandArgs(trailingOnly=TRUE)

if (length(args)< 2) {
    stop("Usage: ./plots.R <infile> <infile...>", call.=FALSE)
}

csv <- args[1]
out <- args[2]

data <- read.csv(file=csv, header=TRUE, sep=",")

p <- ggplot(data=data, aes(x=factor(Nodes), y=Tmedian, group=Algo, colour=Algo)) +
    geom_line() +
    geom_point() +
    ggtitle(paste("Blocksize ", data[1,5]))

ggsave(out, plot=p);

#write.table(csv.data, file=stdout(), row.names=FALSE, col.names=TRUE, sep=",")


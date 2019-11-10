#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(tikzDevice)
library(extrafont)
suppressMessages(library(readr))
suppressMessages(library(ggforce))
suppressMessages(library(tidyverse))

print.gg_multiple <- function(x, page, ...) {
  # Get total number of pages
  page_tot <- ggforce::n_pages(repair_facet(x))

  # Get and check the page number to be drawn
  if (!missing(page)) {
    page_2_draw <- page
  } else {
    page_2_draw <- 1:page_tot
  }

  # Prevent issue with repair_facet when page = NULL
  x$facet$params$page <- page_2_draw

  # Begin multiple page ploting
  n_page_2_draw <- length(page_2_draw)

  # Draw all pages
  for (p in seq_along(page_2_draw)) {
    x$facet$params$page <- page_2_draw[p]
    ggplot2:::print.ggplot(x = repair_facet(x), ...)

  }

  # Prevent ggforce from droping multiple pages value
  x$facet$params$page <- page_2_draw
}

# Fix for ggforce facet_wrap_paginate
repair_facet <- function(x) {
  if (class(x$facet)[1] == 'FacetWrapPaginate' &&
      !'nrow' %in% names(x$facet$params)) {
    x$facet$params$nrow <- x$facet$params$max_row
  }
  x
}

f <- file("stdin")
df.in <- read_csv(paste(collapse = "\n", readLines(f)), col_names=TRUE, col_types="iii?iciicd")
close(f)

#args = commandArgs(trailingOnly=TRUE)
#
#if (length(args)< 1) {
#    stop("Usage: ./plots.R <infile>", call.=FALSE)
#}
#
#csv <- args[1]
#
#data <- read.csv(file=csv, header=TRUE, sep=",")

cols <- c("Bruck", "Bruck_Mod", "Bruck_interleave")

df.stats <- df.in %>%
        filter(Algo %in% cols) %>%
    group_by(Nodes, Procs, Round, NBytes, Blocksize, Algo, Measurement) %>%
    summarise(median=median(Value, na.rm = T)) %>%
    mutate(
           Percent = median/median[Measurement=="Ttotal"]
          )

cat(format_csv(df.stats))

stop()

plotName <- paste(
                  sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv)),
                  ".pdf", sep="")

#cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right

# bars won't be dodged!
pdf(plotName,paper="a4")

mylimit <- function(x) {
    limits <- c(min(x) - .2, max(x) + .2)
    limits
}


p <- ggplot(data, aes(x=factor(Blocksize), y=Ttotal_speedup, colour=Algo, group=Algo)) +
    #geom_errorbar(aes(ymin=Ttotal_med_lowerCI, ymax=Ttotal_med_upperCI), colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=2) +
    theme_bw() +
    # To use for line and point colors, add
    scale_colour_brewer(type="qal", palette="Paired") +
    scale_y_continuous(breaks=seq(0,5,by=.2),limits=mylimit) +
    xlab("Blocksize") +
    ylab("Speedup")+
    geom_hline(yintercept = 1) +
    #annotate("text", min(the.data$year), 50, vjust = -1, label = "Cutoff")
    # + facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
    facet_wrap_paginate( ~Nodes, ncol=1, nrow=3, scales="free_y", page=NULL)


# Here we add our special class
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


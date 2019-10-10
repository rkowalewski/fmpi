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

args = commandArgs(trailingOnly=TRUE)

if (length(args)< 1) {
    stop("Usage: ./plots.R <infile> <outfile...>", call.=FALSE)
}

csv <- args[1]

data <- read.csv(file=csv, header=TRUE, sep=",")
#head(data)

plotName <- paste(
                  sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv)),
                  ".pdf", sep="")

sel <- "ScatteredPairwiseWaitsomeFlatHandshake"


patterns <-
c("ScatteredPairwiseWaitsomeFlatHandshake([0-9]+)?" = "Ring\\1",
  "ScatteredPairwiseWaitsomeOneFactor([0-9]+)?" = "OneFactor\\1",
  "ScatteredPairwiseOneFactor" = "OneFactor",
  "ScatteredPairwiseFlatHandshake" = "Ring",
  "All2AllMortonZSource" = "ZOrderSource",
  "All2AllMortonZDest" = "ZOrderDest",
  "All2AllNaive" = "Naive"
)


#top2 <- data%>% filter(Algo != "AlltoAll" & Algo != "All2AllMorton") %>%
#    group_by(Procs, Blocksize) %>% slice(1:2) %>% ungroup()


naive <- data %>% filter(Algo == "All2AllNaive")

a2a <- data %>% filter(Algo == "AlltoAll" | Algo == "All2AllMortonZSource" | Algo=="All2AllMortonZDest")

selected <- rbind(a2a, naive) %>%
                        mutate(Algo = str_replace_all(Algo, patterns)) %>%
                        arrange(Procs, Blocksize, Ttotal_median)

#head(selected)
#i1 <- selected$Algo =='MortonOrder'
#selected[i1, 9] <- selected[i1, 9] * 1.1
#head(selected)




#selected <- rbind(top2, a2a) %>% arrange(Procs, Blocksize, Ttotal_median) %>%
#                        mutate(Algo = str_replace_all(Algo, patterns))

#selected <- a2a %>% arrange(Procs, Blocksize, Ttotal_median)

#cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# The errorbars overlapped, so use position_dodge to move them horizontally
pd <- position_dodge(0.1) # move them .05 to the left and right

# bars won't be dodged!
#pdf(plotName,paper="a4")
pdf(plotName)

mylimit <- function(x) {
    limits <- c(min(x) - .2, max(x) + .2)
    limits
}

maxSpeedup <- round(max(selected$Ttotal_speedup), digits=1)
minSpeedup <- round(min(selected$Ttotal_speedup), digits=1)

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 45)

labelfn <- function(value){
  return(paste(value, "Ranks", sep=" "))
}

p <- ggplot(selected, aes(x=factor(Blocksize), y=Ttotal_speedup, colour=Algo, group=Algo)) +
    #geom_errorbar(aes(ymin=Ttotal_med_lowerCI, ymax=Ttotal_med_upperCI), colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=2,aes(shape=Algo)) +
    #theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    theme +
    xlab("Blocksize") +
    ylab("Speedup") +
    # To use for line and point colors, add
    scale_colour_brewer(type="qal", palette="Paired") +
    scale_y_continuous(breaks=seq(minSpeedup, maxSpeedup, by=.2),  limits=mylimit)+
    #facet_zoom(xy = Nodes <= 32, horizontal=FALSE)
    facet_wrap_paginate( ~Procs, ncol=2, nrow=2, scales="free_y", page=NULL, labeller=labeller(Procs=labelfn))


# Here we add our special class
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))
suppressMessages(library(ggforce))

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
    stop("Usage: ./plots.R <infile>", call.=FALSE)
}

csv <- args[1]

df.in <- read_csv(csv, col_names=TRUE, col_types="iii?iccdd")

df.bar <- df.in %>% filter(Measurement != "Ttotal")
df.pnt <- df.in %>% filter(Measurement == "Ttotal")

basename <- sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(csv))

plotName <- paste0(basename,".pdf")

# bars won't be dodged!
pdf(plotName,paper="a4")

ngroups <- nrow(df.bar %>% distinct(Blocksize))
nrow <- ceiling(ngroups/3)

theme <- theme_bw()
# change xaxis text
theme$axis.text.x <- element_text(angle = 45)

my_labeller <- label_bquote(
  cols = .(Blocksize) / .(Nodes)
)

# df.pnt, aes(x=Algo, y=median, group=Nodes, )
p <- ggplot(data=df.bar) +
  geom_bar(aes(y = median, x = Algo, fill = Measurement), stat="identity",
           position='stack', alpha=0.8) +
    # To use for line and point colors, add
    scale_fill_brewer(type="qal", palette="Paired") +
  theme +
  facet_wrap_paginate(Blocksize~Nodes, ncol=3, nrow=3,
                      scales="free_y", page=NULL, labeller = my_labeller)

# other labeller are:
# label_wrap_gen(multi_line=FALSE)
#

# Here we add our special class
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


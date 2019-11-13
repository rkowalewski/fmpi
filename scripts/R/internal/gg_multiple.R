#!/usr/bin/env Rscript --vanilla

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


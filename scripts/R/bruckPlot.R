#!/usr/bin/env Rscript

suppressMessages(library(tidyverse))
suppressMessages(library(ggforce))

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
if(!exists("print.gg_multiple", mode="function")) {
    source(paste0(dirname(thisFile()), "/internal/", "gg_multiple.R"))
}
class(p) <- c('gg_multiple', class(p))

#print(p, page=2)
print(p)
dev.off()


#!/usr/bin/env Rscript --vanilla

remove_extension <- function(file) {
    sub(pattern = "(.*)\\..*$", replacement = "\\1", basename(file))
}

# Convert str input to boolean
str2bool = function(input_str)
{
  if(input_str == "0")
  {
    input_str = FALSE
  }
  else if(input_str == "1")
  {
    input_str = TRUE
  }
  return(input_str)
}

util.str_is_empty <- function(str) {
    return(grepl("^\\s*$", str))
}

util.read_csv <- function(path, col_names, col_types) {
    print(paste0("--reading file: ", path))

    if (path == "-") {
        con <- file("stdin")
        df <-  read_csv(paste(collapse = "\n", readLines(con)), col_names, col_types)
        close(con)
        return(df)
    } else if (file.exists(path)) {
        return(read_csv(path, col_names, col_types))
    } else {
        stop(paste0("-- [ERROR] cannot read file: ", path))
    }
}

util.write_csv <- function(df, path) {
    print(paste0("--writing file: ", path))

    if (path == "-") {
        writeLines(format_csv(df), stdout())
    } else {
        write_csv(df, path, na = "NA", append = FALSE, col_names = TRUE,
            quote_escape = "double")
    }
}

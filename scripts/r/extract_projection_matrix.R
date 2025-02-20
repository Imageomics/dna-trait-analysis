# Read in arguments
args <- commandArgs(trailingOnly = TRUE)
input_rds_file <- args[1]
output_csv_file <- args[2]

# Read in .rds file with pca information
h_pca <- readRDS(input_rds_file)

# Write projection matrix to csv
write.csv(h_pca$rotation, output_csv_file, row.names=FALSE)

# Read in arguments
print("Starting R Script")
args <- commandArgs(trailingOnly = TRUE)
input_rds_file <- args[1]
output_csv_file <- args[2]
output_csv_file_center <- args[3]

# Read in .rds file with pca information
print("Reading RDS file:")
h_pca <- readRDS(input_rds_file)

# Write projection matrix to csv
write.csv(h_pca$rotation, output_csv_file, row.names=FALSE)
write.csv(h_pca$center, output_csv_file_center, row.names=FALSE)

# =========================================================================
# 80:20 Stratified Splitting (Training vs. Testing)
# =========================================================================
library(dplyr)
library(splitstackshape)
library(ggplot2)
library(readxl)
library(vcfR)

# Set working directory
setwd("C:/Users/ehtis/OneDrive - New Mexico State University/SUNNY/Research Projects/Mechanical Harvest Projects/genomic prediction/stratified")
getwd()
list.files()

# =========================================================================
# STACK AND COMPILE PHENOTYPIC DATA
# =========================================================================
# import 
pheno22 <- read_excel("pheno2022.xlsx")
pheno23 <- read_excel("pheno2023.xlsx")
pheno25 <- read_excel("pheno2025.xlsx")
# add year as factor 
pheno22$Year <- as.factor("2022")
pheno23$Year <- as.factor("2023")
pheno25$Year <- as.factor("2025")

# this very important, change the "DSFR" to other trait. 
# rest of the pipeline will be same, giving stratified data splitting  
cols_to_keep <- c("geno", "Year", "DSFG")
pheno_master <- rbind(pheno22[, cols_to_keep], 
                      pheno23[, cols_to_keep], 
                      pheno25[, cols_to_keep])
str(pheno_master)
# number of genotypes 
length(unique(pheno_master$geno))
length(unique(pheno_master$Year))


# =========================================================================
# STACK AND COMPILE GENOTYPIC DATA = SNPs (VCF)
# =========================================================================
# import 
# import the geno file (SNPs)  VCF format 
vcf22 <- read.vcfR("geno2022_KNNimp_MAF0.05_Hetro0.1_beagleimp.vcf.vcf", verbose = FALSE)
vcf23 <- read.vcfR("geno2023_KNNimp_MAF0.05_Hetro0.1._beagleimp.vcf.vcf", verbose = FALSE)
vcf25 <- read.vcfR("geno2025_KNNimp_MAF0.05_Hetro0.1._beagleimp.vcf.vcf", verbose = FALSE)

# convert the VCF to numeric format and take transpose   
numeric22 <- extract.gt(vcf22, element = "GT", as.numeric = TRUE)
geno22 <- t(numeric22)
numeric23 <- extract.gt(vcf23, element = "GT", as.numeric = TRUE)
geno23 <- t(numeric23)
numeric25 <- extract.gt(vcf25, element = "GT", as.numeric = TRUE)
geno25 <- t(numeric25)

str(geno22)
str(geno23)
str(geno25)

# Get a completely unique list of markers across all three years
all_unique_snps <- unique(c(colnames(geno22), colnames(geno23), colnames(geno25)))

# This function takes a year's matrix, adds columns for the missing SNPs 
# filled with NA, and ensures perfect column order.
expand_to_union <- function(geno_matrix, target_snps) {
  # Find which markers are missing from this specific matrix
  missing_snps <- setdiff(target_snps, colnames(geno_matrix))
  
  # Create a blank matrix for the missing SNPs filled with NA
  na_matrix <- matrix(NA, 
                      nrow = nrow(geno_matrix), 
                      ncol = length(missing_snps),
                      dimnames = list(rownames(geno_matrix), missing_snps))
  
  # Combine the original data with the new NA columns
  expanded_matrix <- cbind(geno_matrix, na_matrix)
  
  # Force the columns into the exact same master order
  expanded_matrix <- expanded_matrix[, target_snps]
  
  return(expanded_matrix)
}

# expand and align for each year 
geno22_expanded <- expand_to_union(geno22, all_unique_snps)
geno23_expanded <- expand_to_union(geno23, all_unique_snps)
geno25_expanded <- expand_to_union(geno25, all_unique_snps)

# row alignment and vertical stacking 
# align rows to match phenotypic data exactly
geno22_final <- geno22_expanded[pheno22$geno, ]
geno23_final <- geno23_expanded[pheno23$geno, ]
geno25_final <- geno25_expanded[pheno25$geno, ]

# stack the GBS-SNP data for 2022, 2023, 2025
geno_master <- rbind(geno22_final, geno23_final, geno25_final)
str(geno_master)

# rrBLUP's A.mat automatically detects the NAs we introduced and imputes 
# them using the mean of the population for that marker
geno_master <- rrBLUP::A.mat(geno_master, 
                         impute.method = "mean", 
                         return.imputed = TRUE)

# Extract the two components
geno_master_output <- geno_master$A
geno_master_imputed <- geno_master$imputed

# Check: Any remaining NAs? (Should be 0)
missing_count <- sum(is.na(geno_master_imputed))
cat("Remaining NA values in imputed genotype matrix:", missing_count, "\n")

# Check: Row/Column alignment
cat("Imputed Genotype Matrix Dimensions:", nrow(geno_master_imputed), "rows x", ncol(geno_master_imputed), "markers\n")
cat("Kinship Matrix Dimensions:", nrow(geno_master_output), "rows x", ncol(geno_master_output), "cols\n")

# Check: Kinship Diagonal (Should be close to 1, representing self-relationship)
mean_diagonal <- mean(diag(geno_master_output))
cat("Average diagonal value of Kinship Matrix (Self-identity):", round(mean_diagonal, 3), "\n")

# Check: Kinship Off-Diagonal (Should be lower, representing relationships between lines)
off_diag_elements <- geno_master_imputed[lower.tri(geno_master_imputed)]
cat("Average off-diagonal value (Background kinship):", round(mean(off_diag_elements), 3), "\n")
cat("Maximum relationship found between different lines:", round(max(off_diag_elements), 3), "\n")

# heat map to ensure no block anomalies between 2022 and 2023 rows
image(geno_master_output, main = "Kinship Matrix Heatmap", xaxt = "n", yaxt = "n")

# Do not forget to change the dir as per the trait 
dir.create("datasplit/DSFG", showWarnings = FALSE)
list.dirs()
saveRDS(geno_master, "./datasplit/DSFG/geno_master.rds")
saveRDS(geno_master_output, "./datasplit/DSFG/geno_master_output.rds")

# save in excel 
write.csv(geno_master, "./datasplit/DSFG/geno_master.csv", row.names = FALSE)
write.csv(geno_master_output, "./datasplit/DSFG/geno_master_output.csv", row.names = FALSE)

# # =========================================================================
# # PROCESS PHENOTYPIC STRATA (LITERATURE-BASED THRESHOLDS)
# # Reference: https://www.nature.com/articles/s41598-025-94819-z
# # =========================================================================
# 
# lines repeat across years, calculate the overall mean performance per line
# line_pheno_profile <- pheno_master %>%
#  group_by(geno) %>%
#  summarise(Mean_DSRG = mean(DSRG, na.rm = TRUE)) %>%
#  ungroup()
# #Apply strict thresholds from DSR categories
# line_pheno_profile <- line_pheno_profile %>%
#  mutate(Pheno_Strata = case_when(
#    Mean_DSRG <= 2 ~ "Excellent",
#    Mean_DSRG > 2 & Mean_DSRG <= 3 ~ "Good",
#    Mean_DSRG > 3 ~ "Unacceptable",
#    TRUE ~ NA_character_
#  ))
# 
# # # Check the new balanced distribution
# print(table(line_pheno_profile$Pheno_Strata))
# 
# cat("=== counts of how many unique genotypes fell into each category ===\n")
# print(table(line_pheno_profile$Pheno_Strata))
# write.csv(line_pheno_profile, "./datasplit/DSRG/line_pheno_profile.csv", row.names = FALSE)
# 

# 
# # =========================================================================
# # PROCESS PHENOTYPIC STRATA (PERCENTILE-BASED SPLIT FOR BALANCED GROUPS)
# # Trait: Total Yield (TYLD) - Higher values are better (Excellent)
# # =========================================================================
# 
# # Calculate the overall mean performance per line across years
# line_pheno_profile <- pheno_master %>%
#   group_by(geno) %>%
#   summarise(Mean_TYLD = mean(TYLD, na.rm = TRUE)) %>%
#   ungroup()
# 
# # Determine exact percentiles (Bottom 25% and Top 25%)
# # Note: For Yield, the lowest values are bad, highest are excellent
# lower_cutoff <- quantile(line_pheno_profile$Mean_TYLD, 0.25, na.rm = TRUE) # 25th percentile (Low Yield)
# upper_cutoff <- quantile(line_pheno_profile$Mean_TYLD, 0.75, na.rm = TRUE) # 75th percentile (High Yield)
# 
# # Apply percentile thresholds
# line_pheno_profile <- line_pheno_profile %>%
#   mutate(Pheno_Strata = case_when(
#     Mean_TYLD >= upper_cutoff ~ "Excellent",
#     Mean_TYLD > lower_cutoff & Mean_TYLD < upper_cutoff ~ "Good",
#     Mean_TYLD <= lower_cutoff ~ "Unacceptable",
#     TRUE ~ NA_character_
#   ))
# 
# # Check the new balanced distribution
# print(table(line_pheno_profile$Pheno_Strata))
# 
# cat("=== counts of how many unique genotypes fell into each category ===\n")
# print(table(line_pheno_profile$Pheno_Strata))
# #write.csv(line_pheno_profile, "./datasplit/TYLD/line_pheno_profile.csv", row.names = FALSE)

# # =========================================================================
# # PROCESS PHENOTYPIC STRATA (LITERATURE-BASED THRESHOLDS)
# # Reference: https://www.nature.com/articles/s41598-025-94819-z
# # =========================================================================
# lines repeat across years, calculate the overall mean performance per line
line_pheno_profile <- pheno_master %>%
  group_by(geno) %>%
  summarise(Mean_DSFG = mean(DSFG, na.rm = TRUE)) %>%
  ungroup()

# Apply strict thresholds from
line_pheno_profile <- line_pheno_profile %>%
  mutate(Pheno_Strata = case_when(
    Mean_DSFG <= 40 ~ "Low",                      # Low destemming force
    Mean_DSFG > 40 & Mean_DSFG <= 60 ~ "Medium",  # Medium destemming force
    Mean_DSFG > 60 ~ "High",                     # High destemming force
    TRUE ~ NA_character_
  ))

cat("=== counts of how many unique genotypes fell into each category ===\n")
print(table(line_pheno_profile$Pheno_Strata))
write.csv(line_pheno_profile, "./datasplit/DSFG/line_pheno_profile.csv", row.names = FALSE)


# =========================================================================
# ISOLATE THE UNIQUE GENOTYPES FROM PHENO AND GENO DATA
# =========================================================================
# Extract unique line identifiers using pheno master file (across years) 
unique_line_ids <- unique(pheno_master$geno)

# Use match to isolate exactly one row per unique line from the imputed matrix
# This prevents the PCA and K-means clustering from being heavily biased by duplicate rows
geno_unique <- geno_master_imputed[match(unique_line_ids, rownames(geno_master_imputed)), ]

cat("\nUnique Genotype Subset Matrix Dimensions:", nrow(geno_unique), "lines x", ncol(geno_unique), "markers\n")
# Open a connection to the text file
sink("pop_str_report.txt", append = TRUE)
# Print or cat your report text
cat("\nUnique Genotype Subset Matrix Dimensions:", nrow(geno_unique), "lines x", ncol(geno_unique), "markers\n")
#Close the file connection to save changes
sink()

write.csv(geno_unique, "geno_unique_pop_str.csv", row.names = FALSE)


# =========================================================================
# PRINCIPAL COMPONENT ANALYSIS (PCA) & CLUSTERING ON UNIQUE GENOTYPES
# Keep that in mind that PCA will only change if your are working with 
# another population. For this study we have the same population, therefore 
# population structure and clusters remain the same, except the phenotypic 
# variance when the trait is changed. Please see line 29 of the script
# =========================================================================

# Remove monomorphic SNPs (where variance is 0)
geno_clean <- geno_unique[, apply(geno_unique, 2, var, na.rm = TRUE) > 0]

# Scale matrix (mean=0, variance=1) to treat all genetic markers equally
geno_scaled <- scale(geno_clean, center = TRUE, scale = TRUE)

# Run PCA
pca_result     <- prcomp(geno_scaled, rank. = 30, center = FALSE, scale. = FALSE)
total_variance <- ncol(geno_clean) 
pve            <- (pca_result$sdev^2 / total_variance) * 100
pca_df         <- as.data.frame(pca_result$x)

plot(pca_result, type = "lines", main = "PCA Scree Plot")
# Save Scree Plot
# png("scree_plot.png", width = 8, height = 6, units = "in", res = 600)
# plot(pca_result, type = "lines", main = "PCA Scree Plot")
# dev.off()

# Compute & Save Elbow Plot 
n_pcs_for_clustering <- 10
wss <- sapply(2:10, function(k) {
  kmeans(pca_result$x[, 1:n_pcs_for_clustering], centers = k, nstart = 25)$tot.withinss
})

plot(2:10, wss, type = "b", pch = 19,
     xlab = "Number of Clusters (k)", 
     ylab = "Within Sum of Squares",
     main = "Elbow Plot - Choose number of genetic blocks")

# Save what you are currently looking at to a file
# dev.copy(png, filename = "elbow_plot.png", width = 8, height = 6, units = "in", res = 600)
# dev.off() # Closes the file device and resets output back to your screen

# Run K-means clustering (K=5 based on the elbow/scree evaluations)
km_res         <- kmeans(pca_df[, c("PC1", "PC2")], centers = 5, nstart = 25)
pca_df$Group   <- as.factor(km_res$cluster)

# Generate PCA Cluster Plot
pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Group)) +
  geom_point(alpha = 0.7, size = 3.5) +
  scale_color_brewer(palette = "Set1") + 
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),    # Centers the main title
    plot.subtitle = element_text(hjust = 0.5, face = "bold")  # Centers the subtitle
  ) +
  labs(
    title    = "Population Structure",
    subtitle = "K-means clustering (K=5)",
    x        = paste0("PC1 (", round(pve[1], 2), "%)"),
    y        = paste0("PC2 (", round(pve[2], 2), "%)"),
    color    = "Cluster"
  )
print(pca_plot)

# Save PCA plot 
# ggsave("pca_clusters_plot.png", plot = pca_plot, width = 10, height = 6, dpi = 600)
# save genotypes by cluster, their names and respective clusters 
genotype_clusters <- data.frame(
  Genotype_ID = rownames(pca_df),
  Cluster     = pca_df$Group,
  stringsAsFactors = FALSE
)
# write.csv(genotype_clusters, "genotype_by_cluster.csv", row.names = FALSE)

# =========================================================================
# STRATIFIED GENETIC BLOCKS
# =========================================================================
# Extract the top 10 principal components to perform clustering
n_pcs_for_clustering <- 10
pc_scores <- pca_result$x[, 1:n_pcs_for_clustering]

# Run K-means using your selected elbow-point threshold of 5 blocks
set.seed(789) # Ensures stable, identical cluster generation across runs
n_blocks <- 5 # change based on the PCA and Elbow plots  
clusters <- kmeans(pc_scores, centers = n_blocks, nstart = 50)

# -------------------------------------------------------------------------
# Construct Joint Fused Blocks & Execute 80:20 Partitioning
# -------------------------------------------------------------------------
stratification_frame <- data.frame(
  geno = unique_line_ids,
  Gen_Block = as.factor(clusters$cluster)
) %>%
  left_join(line_pheno_profile, by = "geno") %>%
  # Physically bind the SNP family group to the field phenotypic tier
  mutate(Joint_Strat_Block = paste0("G", Gen_Block, "_P", Pheno_Strata))

# Filter out singleton blocks to eliminate allocation errors in stratified()
block_counts <- table(stratification_frame$Joint_Strat_Block)
valid_blocks <- names(block_counts[block_counts > 1])
stratification_frame <- stratification_frame %>% 
  filter(Joint_Strat_Block %in% valid_blocks)

# Execute Proportional Stratified Sampling on unique lines
set.seed(101) 
unique_train_lines <- stratified(stratification_frame, group = "Joint_Strat_Block", size = 0.8, replace = FALSE)
unique_test_lines  <- anti_join(stratification_frame, unique_train_lines, by = "geno")

# -------------------------------------------------------------------------
# Map Lines Back to Full Multi-Year Spatial Rows
# -------------------------------------------------------------------------
pheno_train_final <- pheno_master %>% filter(geno %in% unique_train_lines$geno)
pheno_test_final  <- pheno_master %>% filter(geno %in% unique_test_lines$geno)

# Pull matching row matrices from the full historical database
geno_train_final  <- geno_master_imputed[pheno_train_final$geno, ]
geno_test_final   <- geno_master_imputed[pheno_test_final$geno, ]

# Sub-partition the master Kinship Matrix (A) for mixed-model cross validation
K_train_final     <- geno_master_output[pheno_train_final$geno, pheno_train_final$geno]
K_test_final      <- geno_master_output[pheno_test_final$geno, pheno_test_final$geno]

# -------------------------------------------------------------------------
# Export Target Population Data
# -------------------------------------------------------------------------
# dir.create("datasplit/DSRG", showWarnings = FALSE)
saveRDS(pheno_train_final, "datasplit/DSFG/pheno_train_final.rds")
saveRDS(pheno_test_final,  "datasplit/DSFG/pheno_test_final.rds")
saveRDS(geno_train_final, "datasplit/DSFG/geno_train_final.rds")
saveRDS(geno_test_final,  "datasplit/DSFG/geno_test_final.rds")
saveRDS(K_train_final,    "datasplit/DSFG/K_train_final.rds")

# save csv 
write.csv(pheno_train_final, "datasplit/DSFG/pheno_train_final.csv", row.names = FALSE)
write.csv(pheno_test_final,  "datasplit/DSFG/pheno_test_final.csv",  row.names = FALSE)
write.csv(geno_train_final,  "datasplit/DSFG/geno_train_final.csv",  row.names = FALSE)
write.csv(geno_test_final,   "datasplit/DSFG/geno_test_final.csv",   row.names = FALSE)
write.csv(K_train_final,     "datasplit/DSFG/K_train_final.csv",     row.names = FALSE)

cat("\n=== STRATIFICATION AND DATA SPLITTING COMPLETE ===\n")
cat("Training Population Size :", length(unique(pheno_train_final$geno)), "lines (", nrow(pheno_train_final), "seasonal rows)\n")
cat("Testing Population Size  :", length(unique(pheno_test_final$geno)), "lines (", nrow(pheno_test_final), "seasonal rows)\n")


# =========================================================================
# VISUALIZE PHENOTYPIC DISTRIBUTION ACROSS SUBPOPULATIONS
# =========================================================================
# Map cluster assignments back to the master phenotypic dataset
pheno_subpop <- pheno_master %>%
  left_join(genotype_clusters %>% rename(geno = Genotype_ID, Subpop = Cluster), by = "geno")

# Boxplot of phenotypic variance across subpopulations
p_box <- ggplot(pheno_subpop, aes(x = factor(Subpop), y = DSFG, fill = factor(Subpop))) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16) +
  theme_minimal() +
  labs(
    title = "Phenotypic Distribution (DSFG) by Subpopulation",
    x     = "Subpopulation (Genetic Cluster)",
    y     = "Destemming Force for Green Fruit (DSFG)",
    fill  = "Subpop"
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Density distribution per subpopulation
p_density <- ggplot(pheno_subpop, aes(x = DSFG, fill = factor(Subpop))) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Subpop, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Trait Density Plots per Subpopulation",
    x     = "Destemming Force for Green Fruit (DSFG)",
    y     = "Density"
  ) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# Display and save plots
print(p_box)
print(p_density)

dir.create("datasplit/DSFG/subpop", showWarnings = FALSE)
ggsave("./datasplit/DSFG/subpop/DSFG_subpop_boxplot.png", plot = p_box, width = 8, height = 6, dpi = 600)
ggsave("./datasplit/DSFG/subpop/DSFG_subpop_density.png", plot = p_density, width = 10, height = 6, dpi = 600)


# =========================================================================
# EXPORT INDIVIDUAL SUBPOPULATION DATASETS (RDS & CSV)
# =========================================================================
subpop_dir <- "./datasplit/DSFG/subpop/"

# Subset and save Pheno, Geno, and Kinship matrices for each subpopulation
for (k in 1:max(as.numeric(genotype_clusters$Cluster))) {
  
  # Lines belonging to cluster k
  lines_k <- genotype_clusters$Genotype_ID[genotype_clusters$Cluster == k]
  
  # Filter Phenotype, Genotype, and Kinship matrices
  pheno_k <- pheno_master %>% filter(geno %in% lines_k)
  geno_k  <- geno_master_imputed[rownames(geno_master_imputed) %in% pheno_k$geno, , drop = FALSE]
  K_k     <- geno_master_output[rownames(geno_master_output) %in% pheno_k$geno, 
                                colnames(geno_master_output) %in% pheno_k$geno, drop = FALSE]
  
  # --- Export RDS Files (Best for R workflows) ---
  saveRDS(pheno_k, paste0(subpop_dir, "pheno_subpop_", k, ".rds"))
  saveRDS(geno_k,  paste0(subpop_dir, "geno_subpop_", k, ".rds"))
  saveRDS(K_k,     paste0(subpop_dir, "K_subpop_", k, ".rds"))
  
  # --- Export CSV Files (Best for Python, Excel, or tabular viewing) ---
  write.csv(pheno_k, paste0(subpop_dir, "pheno_subpop_", k, ".csv"), row.names = FALSE)
  write.csv(geno_k,  paste0(subpop_dir, "geno_subpop_", k, ".csv"), row.names = TRUE)
  write.csv(K_k,     paste0(subpop_dir, "K_subpop_", k, ".csv"), row.names = TRUE)
  
  cat(paste0("Exported Subpop ", k, " (RDS & CSV): ", length(unique(pheno_k$geno)), " unique lines\n"))
}


# =========================================================================
# TARGETED FORWARD PREDICTION & AUTO-GENERATED EXPERIMENT LOG
# =========================================================================

library(dplyr)

# -------------------------------------------------------------------------
# DEFINE PARAMETERS & OUTPUT PATH
# -------------------------------------------------------------------------
trait_name        <- "DSFG"
train_test <- 4  # The subpopulation held out for testing

# Clean output path using test_train
output_dir <- paste0("./datasplit/", trait_name, "/test_train/")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------------------------------------------------------
# ISOLATE TRAIN AND TEST LINE IDENTIFIERS
# -------------------------------------------------------------------------
all_subpops   <- sort(unique(as.numeric(genotype_clusters$Cluster)))
train_subpops <- setdiff(all_subpops, train_test)

test_lines  <- genotype_clusters$Genotype_ID[genotype_clusters$Cluster == train_test]
train_lines <- genotype_clusters$Genotype_ID[genotype_clusters$Cluster != train_test]

# -------------------------------------------------------------------------
# SUBPARTITION DATASETS
# -------------------------------------------------------------------------
pheno_train <- pheno_master %>% filter(geno %in% train_lines)
pheno_test  <- pheno_master %>% filter(geno %in% test_lines)

geno_train <- geno_master_imputed[rownames(geno_master_imputed) %in% pheno_train$geno, , drop = FALSE]
geno_test  <- geno_master_imputed[rownames(geno_master_imputed) %in% pheno_test$geno, , drop = FALSE]

K_train <- geno_master_output[rownames(geno_master_output) %in% pheno_train$geno, 
                              colnames(geno_master_output) %in% pheno_train$geno, drop = FALSE]
K_test  <- geno_master_output[rownames(geno_master_output) %in% pheno_test$geno, 
                              colnames(geno_master_output) %in% pheno_test$geno, drop = FALSE]

# -------------------------------------------------------------------------
# EXPORT RDS AND CSV FILES
# -------------------------------------------------------------------------
# RDS Files
saveRDS(pheno_train, paste0(output_dir, "pheno_train.rds"))
saveRDS(pheno_test,  paste0(output_dir, "pheno_test.rds"))
saveRDS(geno_train,  paste0(output_dir, "geno_train.rds"))
saveRDS(geno_test,   paste0(output_dir, "geno_test.rds"))
saveRDS(K_train,     paste0(output_dir, "K_train.rds"))
saveRDS(K_test,      paste0(output_dir, "K_test.rds"))

# CSV Files
write.csv(pheno_train, paste0(output_dir, "pheno_train.csv"), row.names = FALSE)
write.csv(pheno_test,  paste0(output_dir, "pheno_test.csv"),  row.names = FALSE)
write.csv(geno_train,  paste0(output_dir, "geno_train.csv"),  row.names = TRUE)
write.csv(geno_test,   paste0(output_dir, "geno_test.csv"),   row.names = TRUE)
write.csv(K_train,     paste0(output_dir, "K_train.csv"),     row.names = TRUE)
write.csv(K_test,      paste0(output_dir, "K_test.csv"),      row.names = TRUE)

# -------------------------------------------------------------------------
# AUTOMATICALLY GENERATE EXPERIMENT METADATA LOG
# -------------------------------------------------------------------------
log_file <- paste0(output_dir, "README_split_metadata.txt")

log_text <- c(
  "=========================================================================",
  "               GENOMIC PREDICTION EXPERIMENT SPLIT METADATA               ",
  "=========================================================================",
  paste0("Execution Timestamp : ", Sys.time()),
  paste0("Target Trait        : ", trait_name),
  "-------------------------------------------------------------------------",
  "POPULATION ALLOCATION LOGIC:",
  paste0("  • Testing Subpopulation (Holdout) : Subpop ", train_test),
  paste0("  • Training Subpopulations          : Subpop(s) ", paste(train_subpops, collapse = ", ")),
  "-------------------------------------------------------------------------",
  "SAMPLE SIZE BREAKDOWN:",
  paste0("  • Training Set Size : ", length(unique(pheno_train$geno)), " unique genotypes (", nrow(pheno_train), " seasonal observation rows)"),
  paste0("  • Testing Set Size  : ", length(unique(pheno_test$geno)), " unique genotypes (", nrow(pheno_test), " seasonal observation rows)"),
  "-------------------------------------------------------------------------",
  "EXPORTED FILES IN THIS DIRECTORY:",
  "  • pheno_train.rds / .csv | pheno_test.rds / .csv",
  "  • geno_train.rds  / .csv | geno_test.rds  / .csv",
  "  • K_train.rds     / .csv | K_test.rds     / .csv",
  "-------------------------------------------------------------------------",
  "TEST SET GENOTYPE IDENTIFIERS:",
  paste(test_lines, collapse = ", "),
  "========================================================================="
)

writeLines(log_text, con = log_file)

cat("\nSplit complete! Data and metadata notebook saved to:\n", output_dir, "\n")





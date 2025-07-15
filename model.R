# Load necessary libraries
library(dplyr)      # For data manipulation
library(ggplot2)    # For visualization
library(cluster)    # For clustering algorithms and evaluation
library(factoextra) # For visualizing clustering results
library(NbClust)    # For helping determine the number of clusters

# Load the data
country_data <- readr::read_csv(".\\data\\Country-data.csv")

# shuffling the dataset
set.seed(30) # FOR REPRODUCIBILITY
country_data_shuffle <- country_data[sample(nrow(country_data)), ]

# Store country names for later use
country_names <- country_data$country
country_data <- country_data %>% select(-country) # Remove country column

print("\nSummary statistics of the data:")
print(summary(country_data))

# Scaling
# Data Preprocessing: Scaling
scaled_country_data <- scale(country_data)

# Convert the scaled data back to a data frame (for easier handling later)
scaled_country_data <- as.data.frame(scaled_country_data)

print("\nSummary statistics of the scaled data:")
print(summary(scaled_country_data))

# --- Determining the Optimal Number of Clusters (k) for K-Means ---
wss <- numeric(20)
for (i in 1:20) {
   kmeans_result <- kmeans(scaled_country_data, centers = i, nstart = 10)
   wss[i] <- sum(kmeans_result$withinss)
}

# Now, the elbow plot should work correctly with this 'wss' vector
elbow_plot <- ggplot(data.frame(clusters = 1:20, WSS = wss), aes(x = clusters, y = WSS)) +
   geom_line() +
   geom_point() +
   geom_vline(xintercept = 3, linetype = "dashed", color = "red") + # Indicating k=3
   labs(title = "Elbow Method for Optimal k",
        x = "Number of Clusters (k)",
        y = "Within-Cluster Sum of Squares (WSS)") +
   theme_minimal()

print(elbow_plot)

# Silhouette Analysis
silhouette_scores <- c()
for (i in 2:10) {
   kmeans_result <- kmeans(scaled_country_data, centers = i, nstart = 10)
   silhouette_avg <- silhouette(kmeans_result$cluster, dist(scaled_country_data)) %>%
      summary() %>%
      .$avg.width
   silhouette_scores <- c(silhouette_scores, silhouette_avg)
}

silhouette_plot <- ggplot(data.frame(clusters = 2:10, Silhouette = silhouette_scores), aes(x = clusters, y = Silhouette)) +
   geom_line() +
   geom_point() +
   # geom_vline(xintercept = 3, linetype = "dashed", color = "red") + # Indicating k=3
   labs(title = "Silhouette Analysis for Optimal k",
        x = "Number of Clusters (k)",
        y = "Average Silhouette Width") +
   theme_minimal()

print(silhouette_plot)

# 3. Using the NbClust package (provides multiple indices)
nbclust_result <- NbClust(scaled_country_data, min.nc = 2, max.nc = 5, method = "kmeans")
print("NbClust Results:")
print(nbclust_result$Best.nc)

# Applying the K-Means Algorithm with k=3
set.seed(123) # For reproducibility
kmeans_result <- kmeans(scaled_country_data, centers = 3, nstart = 20)

# Cluster assignments for each country
cluster_assignments <- kmeans_result$cluster
print("\nK-Means Cluster Assignments:")
print(cluster_assignments)

# Cluster centers (mean values of features for each cluster)
cluster_centers <- as.data.frame(kmeans_result$centers)
print("\nK-Means Cluster Centers (Scaled Data):")
print(cluster_centers)

# Add cluster assignments back to the original (unscaled) data for interpretation
clustered_data <- cbind(country_data, Cluster = as.factor(cluster_assignments))
clustered_data <- cbind(data.frame(country = country_names), clustered_data)
# Rename the the cluster column to remark
clustered_data <- clustered_data |> rename(Remarks = Cluster)
# Adding a categorical remark
clustered_data <- clustered_data %>%
   mutate(Remarks = case_when(
      Remarks == 1 ~ "Low Priority",
      Remarks == 2 ~ "Mid Priority",
      Remarks == 3 ~ "High Priority"
   ))

# Save as a csv file for deployment.
write.csv(clustered_data, ".\\data\\Country-data-clustered.csv")
# Interpreting the Clusters
# We need to analyze the cluster centers to understand the characteristics of each cluster in terms of the original features. Let's unscale the
# cluster centers to make them more interpretable.

# unscale data
unscale <- function(scaled_country_data, original_mean, original_sd) {
   return(scaled_country_data * original_sd + original_mean)
}

# Get original means and standard deviations
original_means <- apply(country_data, 2, mean)
original_sds <- apply(country_data, 2, sd)

# Unscale the cluster centers
unscaled_cluster_centers <- as.data.frame(matrix(nrow = 3, ncol = ncol(country_data)))
colnames(unscaled_cluster_centers) <- colnames(country_data)

for (i in 1:ncol(country_data)) {
   unscaled_cluster_centers[, i] <- unscale(cluster_centers[, i], original_means[i], original_sds[i])
}

print("\nUnscaled K-Means Cluster Centers (Approximation of Original Values):")

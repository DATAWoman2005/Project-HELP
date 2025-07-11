# Clustering Countries in Direst Need for HELP International using Unsupervised Machine Learning

**Project Goal:** To categorize countries based on socio-economic and health factors to identify those in the direst need of aid, enabling HELP International to strategically allocate \$10 million.

**Data Source:** [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

------------------------------------------------------------------------

## 1. Introduction & Problem Definition

### About HELP International

HELP International is a prominent international humanitarian NGO dedicated to combating poverty and delivering essential amenities and relief to communities in developing and underdeveloped countries, especially during periods of economic hardship, disasters, and natural calamities.

### Problem Statement

HELP International has successfully raised approximately \$10 million. The CEO faces the critical decision of how to deploy these funds most strategically and effectively. As a Data Scientist, the objective is to categorize countries using key socio-economic and health indicators that reflect their overall development status. This categorization will allow the CEO to pinpoint the countries in the direst need of aid, which will be designated as "High Priority" for resource allocation. The final output will recommend these high-priority countries.

### Approach

This project will employ unsupervised machine learning, specifically clustering algorithms, to group countries based on their inherent similarities across the provided features. We will aim to categorize countries into three distinct priority levels: High, Mid, and Low.

------------------------------------------------------------------------

## 2. Data Collection and Initial Exploration

The dataset, sourced from Kaggle, contains various socio-economic and health factors for numerous countries.

### 2.1 Import Libraries

The following R libraries are essential for data manipulation, analysis, and visualization in this project:

``` r
library(dplyr)      # For data manipulation and analysis (e.g., selecting columns)
library(ggplot2)    # For creating high-quality data visualizations (e.g., plots)
library(cluster)    # Provides functions for clustering analysis (e.g., silhouette)
library(factoextra) # Enhances clustering visualization (e.g., fviz_nbclust for elbow/silhouette)
library(NbClust)    # Offers various indices to determine the optimal number of clusters
library(readr)      # For efficient reading of delimited files like CSV
```

### 2.2 Data Loading

The dataset is loaded from the specified CSV file.

``` r
# Load the data
cnt_data <- readr::read_csv(".\\data\\Country-data.csv")

# Store country names separately before removing them for clustering
country_names <- cnt_data$country
country_data <- cnt_data %>% select(-country) # Remove country column for clustering
```

### 2.3 Data Information & Initial Insights

Understanding the raw data's structure and content is the first critical step.

``` r
# Display the first 5 rows to get a quick overview
print("First 5 rows of the dataset:")
head(cnt_data)

# Show the structure of the dataset (column names, data types, and initial values)
print("\nStructure of the dataset:")
str(cnt_data)

# Check for any missing values across the entire dataset
print("\nChecking for missing values:")
sum(is.na(cnt_data))

# Example: Inspecting data for a specific country (e.g., Nigeria)
print("\nData for Nigeria:")
cnt_data[cnt_data$country == "Nigeria", ]
```

**Observation from Initial Exploration:**

-   The dataset contains quantitative variables representing various socio-economic and health indicators.
-   (Add observations regarding `str(cnt_data)` output, e.g., all columns are numeric except `country`.)
-   Crucially, the check for missing values (`sum(is.na(cnt_data))`) should ideally return `0`, indicating a clean dataset, which is common for well-curated public datasets like this. If there were missing values, appropriate imputation or removal strategies would be necessary.

------------------------------------------------------------------------

## 3. Data Preprocessing

Data preprocessing is vital for clustering algorithms to perform effectively.

### 3.1 Why Scaling is Necessary

K-Means and many other distance-based clustering algorithms are highly sensitive to the scale of features. Features with larger numerical ranges (e.g., `income` values in thousands or tens of thousands) would disproportionately influence the distance calculations compared to features with smaller ranges (e.g., `health` expenditure as a percentage, or `child_mort` rates). This can lead to biased clustering results where groups are formed more on magnitude differences rather than true underlying patterns across all indicators.

**Standardization (Z-score normalization)** is used to mitigate this issue. It transforms the data such that each feature has a mean of 0 and a standard deviation of 1. This brings all features to a comparable scale, ensuring each contributes equally to the distance calculations.

### 3.2 Scaling the Data

``` r
# Data Preprocessing: Scaling the numerical features
scaled_data <- scale(country_data)

# Convert the scaled data back to a data frame for easier manipulation
scaled_data <- as.data.frame(scaled_data)

print("\nSummary statistics of the Scaled Data:")
print(summary(scaled_data))
```

**Observation:** After scaling, the mean of each column will be very close to 0, and the standard deviation will be very close to 1, confirming the standardization.

------------------------------------------------------------------------

## 4. Determining the Optimal Number of Clusters (k)

While the problem statement suggests three priority categories, it's best practice to use data-driven methods to inform and validate the choice of $k$. We will use common techniques to identify a suitable number of clusters.

### 4.1 Elbow Method

The Elbow method plots the Within-Cluster Sum of Squares (WSS) against the number of clusters ($k$). WSS measures the compactness of clusters. As $k$ increases, WSS generally decreases because points are closer to their respective centroids. The "elbow" point on the plot, where the rate of decrease in WSS significantly diminishes, is often considered the optimal $k$, as adding more clusters beyond this point provides only marginal gains in compactness.

``` r
# Calculate WSS for a range of k values (1 to 20)
wss <- numeric(20) # Initialize a numeric vector of length 20
for (i in 1:20) {
  kmeans_result_temp <- kmeans(scaled_data, centers = i, nstart = 20) # Use nstart=20 for robustness
  wss[i] <- sum(kmeans_result_temp$withinss)
}

# Plotting the Elbow Method
elbow_plot <- ggplot(data.frame(clusters = 1:20, WSS = wss), aes(x = clusters, y = WSS)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red", size = 0.8) + # Highlight k=3
  labs(title = "Elbow Method for Optimal Number of Clusters",
       x = "Number of Clusters (k)",
       y = "Within-Cluster Sum of Squares (WSS)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) # Center plot title

print(elbow_plot)
```

**Interpretation of Elbow Plot:** (Based on the image you provided earlier) The plot shows a sharp decrease in WSS from $k=1$ to $k=3$. Beyond $k=3$, the rate of decrease significantly lessens, indicating that $k=3$ is a strong candidate for the "elbow" point.

### 4.2 Silhouette Analysis

Silhouette analysis provides a measure of how well each data point fits into its assigned cluster and how poorly it fits into neighboring clusters. The silhouette coefficient ranges from -1 to +1.

-   **+1:** Indicates a strong assignment to its own cluster and well-separated from other clusters.
-   **0:** Suggests the data point is close to the decision boundary between two clusters.
-   **-1:** Implies the data point might be assigned to the wrong cluster.

The average silhouette width across all data points indicates the overall quality of the clustering. A higher average silhouette width is desirable.

``` r
# Calculate average silhouette width for a range of k values (2 to 20 for consistency with elbow)
silhouette_scores <- c()
for (i in 2:20) {
  kmeans_result_temp <- kmeans(scaled_data, centers = i, nstart = 20) # Use nstart=20
  silhouette_avg <- silhouette(kmeans_result_temp$cluster, dist(scaled_data)) %>%
    summary() %>%
    .$avg.width
  silhouette_scores <- c(silhouette_scores, silhouette_avg)
}

# Plotting the Silhouette Analysis
silhouette_plot <- ggplot(data.frame(clusters = 2:20, Silhouette = silhouette_scores), aes(x = clusters, y = Silhouette)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red", size = 0.8) + # Highlight k=3
  labs(title = "Silhouette Analysis for Optimal Number of Clusters",
       x = "Number of Clusters (k)",
       y = "Average Silhouette Width") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) # Center plot title

print(silhouette_plot)
```

**Interpretation of Silhouette Plot:** (Based on the image you provided earlier) The silhouette plot indicated that $k=4$ yielded the highest average silhouette width, with $k=3$ also showing a respectable value. While $k=4$ appears statistically slightly better by this metric, $k=3$ is very close and aligns with the Elbow method's suggestion and the problem's requirement for three distinct priority levels.

### 4.3 NbClust Package Analysis

The `NbClust` package offers a robust approach by providing 30 different indices to determine the optimal number of clusters. It summarizes the recommendations from multiple perspectives, offering a more comprehensive consensus.

``` r
# Using NbClust to get a consensus on optimal k (checking up to k=10)
# Note: For larger datasets or max.nc, this can take some time.
nbclust_result <- NbClust(scaled_data, min.nc = 2, max.nc = 10, method = "kmeans")
print("\nNbClust Results (Best Number of Clusters recommended by various indices):")
print(nbclust_result$Best.nc)
```

**Interpretation of `NbClust` Results:** (Based on the output you provided earlier) The `NbClust` output presented mixed recommendations, but a significant number of indices (e.g., KL, Hartigan, Scott, Marriot, TrCovW, TraceW, Rubin, Ratkowsky, Ball) converged on **3 clusters** as the optimal choice. Other indices suggested 2, 4, or 5 clusters.

### 4.4 Decision on Optimal 'k'

Considering the consistent signal from the Elbow Method, the reasonable silhouette score for $k=3$, and the strong consensus from multiple `NbClust` indices, coupled with the problem requirement to categorize countries into **three priority levels (High, Mid, Low)**, we will proceed with $k=3$ for our K-Means clustering model.

------------------------------------------------------------------------

## 5. Model Building: K-Means Clustering

With $k=3$ chosen, we apply the K-Means algorithm to group the scaled country data.

### 5.1 Applying K-Means

``` r
# Set seed for reproducibility of KMeans results
set.seed(123)

# Apply K-Means clustering with k=3 centers
kmeans_model <- kmeans(scaled_data, centers = 3, nstart = 20) # nstart for robust initialization
```

### 5.2 Examining Cluster Assignments and Centers

After running the algorithm, we extract the cluster assignments for each country and the coordinates of the cluster centroids.

``` r
# Get cluster assignments for each country
cluster_assignments <- kmeans_model$cluster
print("\nK-Means Cluster Assignments (1, 2, or 3 for each country):")
print(head(cluster_assignments)) # Show for the first few countries

# Get cluster centers (centroids) in the scaled feature space
scaled_cluster_centers <- as.data.frame(kmeans_model$centers)
print("\nK-Means Cluster Centers (in Scaled Data space):")
print(scaled_cluster_centers)

# Add cluster assignments to the original (unscaled) data for interpretation
clustered_data <- cbind(country_data, Cluster = as.factor(cluster_assignments))
final_clustered_data <- cbind(data.frame(country = country_names), clustered_data)

print("\nFirst few rows of Original Data with K-Means Cluster Assignments:")
print(head(final_clustered_data))
```

------------------------------------------------------------------------

## 6. Model Interpretation and Labeling

To derive actionable insights, we must interpret what each numerical cluster (1, 2, 3) means in the context of the original data and then assign the "High Priority," "Mid Priority," and "Low Priority" labels.

### 6.1 Unscaling Cluster Centers for Interpretation

Cluster centers obtained from scaled data are difficult to interpret directly. We need to transform them back to the original scale of the features to understand the typical profile of countries within each cluster.

``` r
# Function to unscale data (inverse of the scale() operation)
unscale <- function(scaled_val, original_mean, original_sd) {
  return(scaled_val * original_sd + original_mean)
}

# Get original means and standard deviations from the original 'country_data'
original_means <- apply(country_data, 2, mean)
original_sds <- apply(country_data, 2, sd)

# Create an empty data frame to store unscaled cluster centers
unscaled_cluster_centers <- as.data.frame(matrix(nrow = 3, ncol = ncol(country_data)))
colnames(unscaled_cluster_centers) <- colnames(country_data)

# Loop through each feature to unscale its cluster centers
for (i in 1:ncol(country_data)) {
  unscaled_cluster_centers[, i] <- unscale(scaled_cluster_centers[, i], original_means[i], original_sds[i])
}

print("\nUnscaled K-Means Cluster Centers (Average Feature Values for each Cluster):")
print(unscaled_cluster_centers)
```

### 6.2 Assigning Priority Labels to Clusters

Based on the `unscaled_cluster_centers` (which you provided earlier), we interpret each cluster's profile to assign priority levels.

**Unscaled Cluster Centers (Reference from previous output):**

| Cluster | child_mort | exports | health | imports | income   | inflation | life_expec | total_fer | gdpp     |
|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|:-------|
| **1**   | 5.00       | 58.74   | 8.81   | 51.49   | 45672.22 | 2.67      | 80.13      | 1.75      | 42494.44 |
| **2**   | 21.93      | 40.24   | 6.20   | 47.47   | 12305.60 | 7.60      | 72.81      | 2.31      | 6486.45  |
| **3**   | 92.96      | 29.15   | 6.39   | 42.32   | 3942.40  | 12.02     | 59.19      | 5.01      | 1922.38  |

**Interpretation and Labeling:**

-   **Cluster 1 (Low Priority):**

    -   **Profile:** Very low child mortality, high life expectancy, very high income and GDP per capita, high exports/imports, stable inflation, and low fertility rates.
    -   **Conclusion:** These are highly developed countries with robust economies and excellent health indicators. They are least in need of emergency aid.

-   **Cluster 2 (Mid Priority):**

    -   **Profile:** Moderate child mortality, moderate life expectancy, moderate income and GDP per capita, and moderate inflation. Exports/imports are also in a mid-range.
    -   **Conclusion:** These countries are likely developing nations. While they have made progress, they may still face significant challenges and could benefit from targeted development aid, but not necessarily emergency relief.

-   **Cluster 3 (High Priority):**

    -   **Profile:** Extremely high child mortality, very low life expectancy, very low income and GDP per capita, high inflation, and very high fertility rates. Exports/imports are relatively lower.
    -   **Conclusion:** These countries exhibit severe indicators across health and economic factors, suggesting widespread poverty and significant humanitarian challenges. These are the countries in the direst need of aid.

### 6.3 Assigning Priority Labels to Countries

Now, we can add a new column to our `final_clustered_data` dataframe that directly assigns these human-readable priority labels.

``` r
# Create a mapping from cluster number to priority label
priority_map <- c("Low Priority", "Mid Priority", "High Priority") # Assuming cluster 1=Low, 2=Mid, 3=High

# Add the priority label column to the clustered data
final_clustered_data$Priority <- factor(
  final_clustered_data$Cluster,
  levels = c(1, 2, 3), # Ensure correct ordering of levels
  labels = priority_map
)

print("\nFirst few rows of Data with Assigned Priority Labels:")
print(head(final_clustered_data))
```

------------------------------------------------------------------------

## 7. Model Evaluation (Internal Metrics)

For unsupervised learning, internal evaluation metrics are used to assess the quality of the clustering structure itself, as there are no true labels to compare against.

### 7.1 Silhouette Score Review

We already calculated the average silhouette width during the determination of optimal $k$. We can also visualize the silhouette plot for the chosen $k=3$ to inspect individual point's silhouette values.

``` r
# Generate the silhouette plot for k=3
silhouette_k3 <- silhouette(kmeans_model$cluster, dist(scaled_data))
fviz_silhouette(silhouette_k3) +
  labs(title = "Silhouette Plot for K-Means (k=3)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
```

**Interpretation:** This plot shows the silhouette coefficient for each data point within its cluster. A high average silhouette width (closer to 1) indicates good separation. Points with low or negative silhouette values might be misclassified or lie between clusters.

### 7.2 Other Internal Metrics (e.g., Davies-Bouldin Index, Dunn Index)

While `NbClust` already considers many, you could individually calculate metrics like the Davies-Bouldin index (lower is better, indicating better separation and compactness) or Dunn index (higher is better, indicating better separation and compactness) using packages like `clusterSim` or by directly using `cluster` output.

``` r
# Example for Davies-Bouldin Index (requires 'clusterSim' or manual calculation)
# install.packages("clusterSim")
# library(clusterSim)
# dbi_score <- index.DB(x = scaled_data, cl = kmeans_model$cluster, centrotypes = "centroids")$DB
# print(paste("Davies-Bouldin Index for k=3:", dbi_score))
```

------------------------------------------------------------------------

## 8. Code Testing and Recommendations

### 8.1 Random Country Check

To test our categorization, we can randomly select a few countries and see which priority category they fall into based on our model.

``` r
# Set seed for reproducibility of random sampling
set.seed(456)

# Randomly select 5 countries
random_countries_indices <- sample(1:nrow(final_clustered_data), 5)
random_countries_check <- final_clustered_data[random_countries_indices, c("country", "Priority")]

print("\nRandom Sample of Countries and their Assigned Priority:")
print(random_countries_check)
```

**Evaluation of Random Countries:** You would manually inspect these countries' original data (e.g., child mortality, income) and cross-reference them with the cluster characteristics to see if the assigned priority makes intuitive sense. For example, if a country known for high development (e.g., Australia) falls into "Low Priority," or a country known for significant challenges (e.g., Afghanistan) falls into "High Priority," it validates the model's logic.

### 8.2 Countries for CEO's Focus (High Priority)

Finally, we list all countries identified as "High Priority" for HELP International's immediate attention.

``` r
# Filter for High Priority countries
high_priority_countries <- final_clustered_data %>%
  filter(Priority == "High Priority") %>%
  select(country, child_mort, income, life_expec, gdpp, Priority) %>%
  arrange(child_mort) # Order by child mortality to see the most severe first

print("\n--- Countries for CEO's Immediate Focus (High Priority) ---")
print(high_priority_countries)

# You can also get the number of countries in each priority group
print("\nNumber of Countries in Each Priority Group:")
print(table(final_clustered_data$Priority))
```

**Recommendation to CEO:** Based on the K-Means clustering analysis, the countries categorized under "High Priority" are those exhibiting the most critical socio-economic and health indicators (highest child mortality, lowest life expectancy, lowest income, etc.). These are the nations where HELP International's \$10 million fund can have the most significant and immediate impact in alleviating dire humanitarian needs. The list of these specific countries is provided above. Further in-depth analysis on these identified countries could refine the allocation strategy.

------------------------------------------------------------------------

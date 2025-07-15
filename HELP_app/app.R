# app.R

#  1. Import Libraries 
library(shiny)
library(dplyr)      # for data manipulation and analysis
library(ggplot2)    # for data visualization
library(moments)    # to calculate skewness
library(readr)      # for efficient reading of delimited files like CSV
library(DT)         # For interactive data tables
library(leaflet)    # For interactive maps
library(sf)         # For spatial data manipulation
library(rnaturalearth) # For world map data
library(rnaturalearthdata)
library(countrycode) # For robust country name matching
library(plotly)     # For interactive ggplot2 plots (hover effects)
library(cluster)    # For silhouette analysis
library(NbClust)    # For optimal k determination
library(purrr)      # For functional programming (map_dfr)

# Disable scientific notation globally for better readability of numbers
options(scipen = 999)

#  2. Data Collection (Load data once for the app) 
# IMPORTANT: Ensure 'Country-data-clustered.csv' and 'data-dictionary.csv'
# are in the same directory as this app.R file, or update the paths.
# The clustered data now includes the 'Remarks' column
cnt_data_raw <- readr::read_csv("Country-data-clustered.csv")

# Ensure 'Remarks' is a factor with specific levels for consistent plotting
if ("Remarks" %in% colnames(cnt_data_raw)) {
   cnt_data_raw$Remarks <- factor(cnt_data_raw$Remarks,
                                  levels = c("High Priority", "Mid Priority", "Low Priority"))
} else {
   # If 'Remarks' column is not found (e.g., first run before clustering script),
   # add a placeholder or handle gracefully. For this app, it's expected.
   warning("Remarks column not found in Country-data-clustered.csv. Some features may not work as expected.")
}

#  Global Data Preparation for Map and Model 
# Get world map data
world_map <- ne_countries(scale = "medium", returnclass = "sf")

# Clean and standardize country names in cnt_data_raw for matching
# Using ISO3c codes for robust matching
cnt_data_processed <- cnt_data_raw %>%
   mutate(iso_a3 = countrycode(country, "country.name", "iso3c"))

# Join your data with the map data
# Perform a left join to keep all map polygons and add your data where matches are found
map_data_joined <- world_map %>%
   left_join(cnt_data_processed, by = c("iso_a3" = "iso_a3"))

# Filter out countries from the map that are not in your dataset or have no data
map_data_filtered <- map_data_joined %>%
   filter(!is.na(country))

# Get list of numerical features for map and plot dropdowns
numeric_cols <- names(select_if(cnt_data_processed, is.numeric))
# Exclude 'iso_a3' if it was created and is numeric
numeric_cols <- numeric_cols[numeric_cols != "iso_a3"]

# Prepare data for K-Means (excluding country and Remarks for clustering)
country_data_for_clustering <- cnt_data_raw %>%
   select(-country, -Remarks) # Exclude non-numeric and target columns

# Store original means and sds for unscaling cluster centers
original_means <- apply(country_data_for_clustering, 2, mean)
original_sds <- apply(country_data_for_clustering, 2, sd)

# Function to unscale data (inverse of the scale() operation)
unscale <- function(scaled_val, original_mean, original_sd) {
   return(scaled_val * original_sd + original_mean)
}

#  Define UI 
ui <- fluidPage(
   # Application title
   titlePanel("Interactive Country Data Analysis: Socio-Economic & Health Factors"),
   
   # Main content layout
   tabsetPanel(
      #  Tab 1: Data Information 
      tabPanel("Data Information",
               h3("Full Dataset Overview"),
               p("This table displays the complete dataset with socio-economic, health factors, and assigned priority. It allows for interactive filtering and searching."),
               br(),
               DT::dataTableOutput("full_data_table"), # The full data table Interactive table
               br()
      ),
      
      #  Tab 2: Key Distributions 
      tabPanel("Key Distributions",
               h3("Understanding Feature Distributions"),
               p("Explore the distribution of individual socio-economic and health indicators. You can select a feature and adjust histogram bins, or view its distribution across priority groups."),
               br(),
               fluidRow(
                  column(6,
                         selectInput("dist_feature", "Select Feature:", choices = numeric_cols, selected = "gdpp")
                  ),
                  column(6,
                         sliderInput("hist_bins", "Number of Bins for Histogram:", min = 5, max = 50, value = 30)
                  )
               ),
               tabsetPanel(
                  tabPanel("Histogram",
                           plotOutput("selected_feature_hist"),
                           verbatimTextOutput("selected_feature_summary_skew")
                  ),
                  tabPanel("Density Plot",
                           plotOutput("selected_feature_density")
                  ),
                  tabPanel("Distribution by Priority",
                           plotOutput("selected_feature_boxplot")
                  )
               ),
               br(),
               h4("Insights on Common Distributions:"),
               p("•   **GDP per Capita (gdpp):** Typically right-skewed, indicating a few wealthy countries and many with lower GDPs. This highlights significant global economic disparity. Countries on the lower end are often those in dire need."),
               p("•   **Inflation Rates:** Often right-skewed, with most countries having moderate inflation, but some experiencing very high rates (economic instability) or even deflation (negative rates).")
      ),
      
      #  Tab 3: Relationships Between Variables 
      tabPanel("Relationships Between Variables",
               h3("Exploring Bivariate Relationships"),
               p("Investigate how different socio-economic and health factors correlate. Hover over points to see country names."),
               br(),
               fluidRow(
                  column(6,
                         selectInput("scatter_x", "Select X-axis Feature:", choices = numeric_cols, selected = "income")
                  ),
                  column(6,
                         selectInput("scatter_y", "Select Y-axis Feature:", choices = numeric_cols, selected = "child_mort")
                  )
               ),
               plotlyOutput("interactive_scatter_plot", height = "600px"),
               verbatimTextOutput("scatter_correlation"),
               br(),
               h4("Key Relationship Insights:"),
               p("•   **GDP per Capita vs. Life Expectancy:** Generally shows a positive correlation, suggesting that higher economic prosperity often leads to better healthcare and living conditions, contributing to longer lives."),
               p("•   **Child Mortality vs. Income Level:** A strong inverse relationship is observed, where child mortality significantly decreases as income level increases. This underscores the impact of economic status on child survival."),
               p("•   **Health Expenditure vs. Life Expectancy:** While a positive trend might exist, the correlation can be weak, indicating that the effectiveness of health spending (how resources are utilized) is as important as the amount spent."),
               p("•   **Exports vs. Imports:** Often positively correlated, implying that countries engaged in more international trade (higher exports) also tend to have higher imports, supporting global supply chains."),
               p("•   **Total Fertility vs. GDP per Capita:** An inverse relationship is common; countries with lower fertility rates often have higher GDP per capita, potentially due to factors like education and development."),
               p("•   **Total Fertility vs. Child Mortality:** A strong positive correlation, highlighting that higher fertility rates are often associated with higher child mortality, reflecting less developed healthcare and family planning.")
      ),
      
      #  Tab 4: Specific Country Insights & Map 
      tabPanel("Specific Country Insights & Map",
               h3("Explore Individual Country Data and Geographical Distribution"),
               p("Visualize the distribution of any feature across countries on the map, colored by its value or by the assigned priority. You can also look up detailed data for specific countries."),
               br(),
               fluidRow(
                  column(6,
                         selectInput("map_feature", "Select Feature for Map:",
                                     choices = c("Remarks", numeric_cols), # Add Remarks as a choice
                                     selected = "Remarks")
                  ),
                  column(6,
                         selectInput("select_country", "Select a Country for Details:", choices = sort(unique(cnt_data_raw$country)))
                  )
               ),
               leafletOutput("country_map", height = "600px"), # Map output
               br(),
               h4("Selected Country Data:"),
               DT::dataTableOutput("selected_country_data_table"), # Display as interactive table
               br()
      ),
      
      #  Tab 5: K-Means Model 
      tabPanel("K-Means Model",
               h3("Interactive K-Means Clustering Exploration"),
               p("Explore the K-Means clustering process. Adjust parameters like the number of clusters (k) and visualize the resulting groups and their characteristics."),
               br(),
               fluidRow(
                  column(4,
                         sliderInput("k_value", "Number of Clusters (k):", min = 2, max = 10, value = 3, step = 1),
                         sliderInput("nstart_value", "Number of Random Starts (nstart):", min = 1, max = 50, value = 20, step = 1)
                  ),
                  column(8,
                         h4("Optimal k Determination Insights"),
                         tabsetPanel(
                            tabPanel("Elbow Plot", plotOutput("model_elbow_plot", height = "300px")),
                            tabPanel("Silhouette Plot", plotOutput("model_silhouette_plot", height = "300px")),
                            tabPanel("NbClust Consensus", verbatimTextOutput("model_nbclust_output"))
                         )
                  )
               ),
               hr(),
               h4("K-Means Clustering Results (Interactive)"),
               fluidRow(
                  column(6,
                         h5("Cluster Visualization (Income vs. Child Mortality)"),
                         plotlyOutput("kmeans_cluster_plot", height = "400px")
                  ),
                  column(6,
                         h5("Dynamic Cluster Centers (Average Unscaled Values)"),
                         DT::dataTableOutput("dynamic_cluster_centers"),
                         h5("Model Performance Metrics:"),
                         verbatimTextOutput("model_metrics")
                  )
               ),
               br(),
               h4("Countries in Each Dynamic Cluster:"),
               uiOutput("countries_in_clusters"),
               hr(),
               h4("Predict Priority for a New/Hypothetical Country"),
               p("Enter values for a country's features to see which priority cluster it would fall into based on the current K-Means model."),
               fluidRow(
                  column(3, numericInput("pred_child_mort", "Child Mortality:", value = 50)),
                  column(3, numericInput("pred_exports", "Exports (% of GDP):", value = 40)),
                  column(3, numericInput("pred_health", "Health (% of GDP):", value = 6)),
                  column(3, numericInput("pred_imports", "Imports (% of GDP):", value = 45))
               ),
               fluidRow(
                  column(3, numericInput("pred_income", "Income:", value = 5000)),
                  column(3, numericInput("pred_inflation", "Inflation:", value = 8)),
                  column(3, numericInput("pred_life_expec", "Life Expectancy:", value = 65)),
                  column(3, numericInput("pred_total_fer", "Total Fertility:", value = 3)),
                  column(3, numericInput("pred_gdpp", "GDP per Capita:", value = 3000))
               ),
               actionButton("predict_button", "Predict Cluster"),
               br(),br(),
               verbatimTextOutput("prediction_output")
      )
   )
)

#  Define server logic 
server <- function(input, output, session) {
   
   # Reactive data for clustering (scaled)
   scaled_country_data_reactive <- reactive({
      scale(country_data_for_clustering)
   })
   
   #  Data Information Outputs 
   output$full_data_table <- DT::renderDataTable({
      DT::datatable(cnt_data_raw,
                    options = list(pageLength = 10, scrollX = TRUE),
                    filter = 'top',
                    rownames = FALSE
      )
   })
   
   output$data_dictionary_output <- DT::renderDataTable({
      DT::datatable(data_dict,
                    options = list(pageLength = 5, scrollX = TRUE),
                    filter = 'top',
                    rownames = FALSE)
   })
   
   #  Key Distributions Outputs 
   output$selected_feature_hist <- renderPlot({
      req(input$dist_feature)
      ggplot(cnt_data_raw, aes_string(x = input$dist_feature)) +
         geom_histogram(bins = input$hist_bins, fill = "skyblue", color = "white") +
         labs(title = paste("Distribution of", input$dist_feature),
              x = input$dist_feature,
              y = "Number of Countries") +
         theme_minimal() +
         theme(plot.title = element_text(hjust = 0.5))
   })
   
   output$selected_feature_density <- renderPlot({
      req(input$dist_feature)
      ggplot(cnt_data_raw, aes_string(x = input$dist_feature)) +
         geom_density(fill = "lightgreen", alpha = 0.7) +
         labs(title = paste("Density Plot of", input$dist_feature),
              x = input$dist_feature,
              y = "Density") +
         theme_minimal() +
         theme(plot.title = element_text(hjust = 0.5))
   })
   
   output$selected_feature_boxplot <- renderPlot({
      req(input$dist_feature, cnt_data_raw$Remarks) # Ensure Remarks exists
      ggplot(cnt_data_raw, aes_string(x = "Remarks", y = input$dist_feature, fill = "Remarks")) +
         geom_boxplot(alpha = 0.7) +
         labs(title = paste("Distribution of", input$dist_feature, "by Priority"),
              x = "Priority Group",
              y = input$dist_feature) +
         scale_fill_manual(values = c("High Priority" = "red", "Mid Priority" = "orange", "Low Priority" = "darkgreen")) +
         theme_minimal() +
         theme(plot.title = element_text(hjust = 0.5))
   })
   
   output$selected_feature_summary_skew <- renderPrint({
      req(input$dist_feature)
      feature_data <- cnt_data_raw[[input$dist_feature]]
      cat("Summary of", input$dist_feature, ":\n")
      print(summary(feature_data))
      cat("\nSkewness of", input$dist_feature, ":", moments::skewness(feature_data), "\n")
   })
   
   #  Relationships Between Variables Outputs 
   output$interactive_scatter_plot <- renderPlotly({
      req(input$scatter_x, input$scatter_y, cnt_data_raw$Remarks)
      
      # Create a reactive data frame for the scatter plot with pre-built tooltip text
      scatter_plot_data <- cnt_data_raw %>%
         mutate(
            tooltip_text_scatter = paste0(
               "Country: ", country,
               "<br>", input$scatter_x, ": ", .data[[input$scatter_x]],
               "<br>", input$scatter_y, ": ", .data[[input$scatter_y]],
               "<br>Priority: ", Remarks
            )
         )
      
      p <- ggplot(scatter_plot_data, aes_string(x = input$scatter_x, y = input$scatter_y, color = "Remarks",
                                                text = "tooltip_text_scatter")) + # Use the pre-built text column
         geom_point(size = 3, alpha = 0.8) +
         scale_color_manual(values = c("High Priority" = "red", "Mid Priority" = "orange", "Low Priority" = "darkgreen")) +
         labs(title = paste(input$scatter_y, "vs.", input$scatter_x),
              x = input$scatter_x,
              y = input$scatter_y) +
         theme_minimal() +
         theme(plot.title = element_text(hjust = 0.5))
      
      ggplotly(p, tooltip = "text") # Make it interactive with plotly, using 'text' for tooltip
   })
   
   output$scatter_correlation <- renderPrint({
      req(input$scatter_x, input$scatter_y)
      x_data <- cnt_data_raw[[input$scatter_x]]
      y_data <- cnt_data_raw[[input$scatter_y]]
      # Handle potential non-numeric data or NA values for correlation
      if (is.numeric(x_data) && is.numeric(y_data)) {
         cor_val <- cor(x_data, y_data, use = "pairwise.complete.obs")
         cat("Correlation between", input$scatter_x, "and", input$scatter_y, ":", round(cor_val, 3), "\n")
      } else {
         cat("Correlation not applicable for selected non-numeric features.\n")
      }
   })
   
   #  Specific Country Insights & Map Outputs 
   
   # Reactive expression for the map data based on selected feature
   reactive_map_data <- reactive({
      req(input$map_feature)
      
      if (input$map_feature == "Remarks") {
         # For categorical 'Remarks'
         pal <- colorFactor(
            palette = c("red", "orange", "darkgreen"), # High, Mid, Low
            domain = map_data_filtered$Remarks,
            levels = c("High Priority", "Mid Priority", "Low Priority"), # Ensure order
            na.color = "#808080"
         )
         feature_values <- map_data_filtered$Remarks
         popup_text <- paste0("<b>", map_data_filtered$country, "</b><br/>",
                              "Priority: ", map_data_filtered$Remarks)
      } else {
         # For numeric features
         feature_values <- map_data_filtered[[input$map_feature]]
         pal <- colorNumeric(
            palette = "viridis",
            domain = feature_values,
            na.color = "#808080"
         )
         popup_text <- paste0("<b>", map_data_filtered$country, "</b><br/>",
                              input$map_feature, ": ", round(feature_values, 2))
      }
      
      map_data_filtered %>%
         mutate(fill_color = pal(feature_values)) %>%
         mutate(popup_text = popup_text)
   })
   
   
   output$country_map <- renderLeaflet({
      leaflet() %>%
         addProviderTiles(providers$CartoDB.Positron) %>%
         setView(lng = 0, lat = 30, zoom = 2)
   })
   
   observe({
      current_map_data <- reactive_map_data()
      # Re-create palette here to ensure it's reactive to the feature selection
      if (input$map_feature == "Remarks") {
         pal <- colorFactor(
            palette = c("red", "orange", "darkgreen"),
            domain = current_map_data$Remarks,
            levels = c("High Priority", "Mid Priority", "Low Priority"),
            na.color = "#808080"
         )
      } else {
         pal <- colorNumeric(
            palette = "viridis",
            domain = current_map_data[[input$map_feature]],
            na.color = "#808080"
         )
      }
      
      leafletProxy("country_map", data = current_map_data) %>%
         clearShapes() %>%
         clearControls() %>%
         
         addPolygons(
            fillColor = ~fill_color,
            weight = 1,
            opacity = 1,
            color = "white",
            dashArray = "3",
            fillOpacity = 0.7,
            highlightOptions = highlightOptions(
               weight = 3,
               color = "#666",
               dashArray = "",
               fillOpacity = 0.7,
               bringToFront = TRUE),
            label = ~country,
            popup = ~popup_text
         ) %>%
         addLegend(pal = pal,
                   values = if (input$map_feature == "Remarks") levels(current_map_data$Remarks) else current_map_data[[input$map_feature]], # Values for legend
                   opacity = 0.7,
                   title = input$map_feature,
                   position = "bottomright")
   })
   
   
   output$selected_country_data_table <- DT::renderDataTable({
      req(input$select_country)
      country_data_display <- cnt_data_raw %>%
         filter(country == input$select_country)
      DT::datatable(country_data_display,
                    options = list(dom = 't', paging = FALSE, searching = FALSE), # Show table, no pagination/search
                    rownames = FALSE)
   })
   
   output$high_fertility_countries <- renderPrint({
      cnt_data_temp <- cnt_data_raw
      cnt_data_temp$fertility_group <- cut(cnt_data_temp$total_fer,
                                           breaks = quantile(cnt_data_temp$total_fer, probs = c(0, 0.33, 0.66, 1)),
                                           labels = c("Low Fertility", "Medium Fertility", "High Fertility"),
                                           include.lowest = TRUE)
      cnt_data_temp[cnt_data_temp$fertility_group == "High Fertility", c("country", "total_fer")]
   })
   
   #  K-Means Model Tab Outputs 
   
   # Reactive K-Means model based on k and nstart sliders
   kmeans_model_reactive <- reactive({
      req(input$k_value, input$nstart_value)
      set.seed(123) # For reproducibility of the random starts
      kmeans(scaled_country_data_reactive(), centers = input$k_value, nstart = input$nstart_value)
   })
   
   # Reactive unscaled cluster centers
   unscaled_cluster_centers_reactive <- reactive({
      kmeans_result <- kmeans_model_reactive()
      scaled_centers <- as.data.frame(kmeans_result$centers)
      
      unscaled_centers <- as.data.frame(matrix(nrow = input$k_value, ncol = ncol(country_data_for_clustering)))
      colnames(unscaled_centers) <- colnames(country_data_for_clustering)
      
      for (i in 1:ncol(country_data_for_clustering)) {
         unscaled_centers[, i] <- unscale(scaled_centers[, i], original_means[i], original_sds[i])
      }
      unscaled_centers
   })
   
   # Optimal k plots (static for now, can be made reactive to max_k_explore if needed)
   output$model_elbow_plot <- renderPlot({
      wss <- numeric(10) # Check up to 10 clusters for display
      for (i in 1:10) {
         kmeans_result_temp <- kmeans(scaled_country_data_reactive(), centers = i, nstart = 10)
         wss[i] <- sum(kmeans_result_temp$withinss)
      }
      ggplot(data.frame(clusters = 1:10, WSS = wss), aes(x = clusters, y = WSS)) +
         geom_line() + geom_point() +
         labs(title = "Elbow Method for Optimal k", x = "Number of Clusters (k)", y = "WSS") +
         theme_minimal() + theme(plot.title = element_text(hjust = 0.5))
   })
   
   output$model_silhouette_plot <- renderPlot({
      silhouette_scores <- c()
      for (i in 2:10) {
         kmeans_result_temp <- kmeans(scaled_country_data_reactive(), centers = i, nstart = 10)
         silhouette_avg <- silhouette(kmeans_result_temp$cluster, dist(scaled_country_data_reactive())) %>%
            summary() %>% .$avg.width
         silhouette_scores <- c(silhouette_scores, silhouette_avg)
      }
      ggplot(data.frame(clusters = 2:10, Silhouette = silhouette_scores), aes(x = clusters, y = Silhouette)) +
         geom_line() + geom_point() +
         labs(title = "Silhouette Analysis for Optimal k", x = "Number of Clusters (k)", y = "Avg. Silhouette Width") +
         theme_minimal() + theme(plot.title = element_text(hjust = 0.5))
   })
   
   output$model_nbclust_output <- renderPrint({
      # NbClust can be computationally intensive, limiting max.nc for interactivity
      nbclust_result <- NbClust(scaled_country_data_reactive(), min.nc = 2, max.nc = input$k_value + 3, method = "kmeans")
      print(nbclust_result$Best.nc)
   })
   
   # Cluster visualization plot
   output$kmeans_cluster_plot <- renderPlotly({
      kmeans_result <- kmeans_model_reactive()
      plot_data <- cnt_data_raw %>%
         mutate(Cluster = as.factor(kmeans_result$cluster)) %>%
         # Pre-build the tooltip text column
         mutate(
            tooltip_text_kmeans = paste0("Country: ", country,
                                         "<br>Income: ", income,
                                         "<br>Child Mort: ", child_mort,
                                         "<br>Cluster: ", Cluster)
         )
      
      # Use the pre-built text column in aes_string
      p <- ggplot(plot_data, aes_string(x = "income", y = "child_mort", color = "Cluster",
                                        text = "tooltip_text_kmeans")) +
         geom_point(size = 3, alpha = 0.8) +
         geom_point(data = unscaled_cluster_centers_reactive(), aes(x = income, y = child_mort),
                    color = "black", shape = 8, size = 5, stroke = 1.5, show.legend = FALSE,
                    inherit.aes = FALSE) + # Ensure centroids don't inherit 'text' aesthetic
         labs(title = paste("K-Means Clusters (k=", input$k_value, ")"),
              x = "Income Per Person", y = "Child Mortality Rate") +
         theme_minimal() +
         theme(plot.title = element_text(hjust = 0.5))
      
      ggplotly(p, tooltip = "text")
   })
   
   output$dynamic_cluster_centers <- DT::renderDataTable({
      DT::datatable(unscaled_cluster_centers_reactive(),
                    options = list(dom = 't', paging = FALSE, searching = FALSE, scrollX = TRUE),
                    rownames = TRUE) # Show cluster numbers as row names
   })
   
   output$model_metrics <- renderPrint({
      kmeans_result <- kmeans_model_reactive()
      cat("Total Within-Cluster Sum of Squares (WSS):", round(sum(kmeans_result$withinss), 2), "\n")
      # Calculate average silhouette width
      if (input$k_value > 1) {
         sil_avg <- silhouette(kmeans_result$cluster, dist(scaled_country_data_reactive())) %>%
            summary() %>% .$avg.width
         cat("Average Silhouette Width:", round(sil_avg, 3), "\n")
      } else {
         cat("Average Silhouette Width: N/A (k must be > 1)\n")
      }
   })
   
   output$countries_in_clusters <- renderUI({
      kmeans_result <- kmeans_model_reactive()
      cluster_assignments <- kmeans_result$cluster
      
      # Create a list of countries for each cluster
      cluster_lists <- purrr::map(1:input$k_value, function(k_num) {
         countries_in_cluster <- cnt_data_raw$country[cluster_assignments == k_num]
         if (length(countries_in_cluster) > 0) {
            tags$div(
               h5(paste("Cluster", k_num, "(", length(countries_in_cluster), "countries):")),
               tags$ul(
                  lapply(countries_in_cluster, tags$li)
               )
            )
         } else {
            tags$div(h5(paste("Cluster", k_num, "(0 countries)")))
         }
      })
      do.call(tagList, cluster_lists) # Combine all cluster divs
   })
   
   # Prediction for new country
   observeEvent(input$predict_button, {
      new_country_data_df <- data.frame(
         child_mort = input$pred_child_mort,
         exports = input$pred_exports,
         health = input$pred_health,
         imports = input$pred_imports,
         income = input$pred_income,
         inflation = input$pred_inflation,
         life_expec = input$pred_life_expec,
         total_fer = input$pred_total_fer,
         gdpp = input$pred_gdpp
      )
      
      # Ensure column order matches original_means/sds for scaling
      # This is crucial for correct scaling
      new_country_data_ordered <- new_country_data_df[, names(original_means)]
      
      # Scale the new data using original data's mean and sd
      scaled_new_country_data <- scale(new_country_data_ordered,
                                       center = original_means,
                                       scale = original_sds)
      
      # Predict cluster by finding the closest centroid
      kmeans_result <- kmeans_model_reactive()
      
      # Calculate Euclidean distance to each centroid
      # Need to ensure scaled_new_country_data is a matrix for dist() if it's a single row
      distances <- apply(kmeans_result$centers, 1, function(center) {
         dist(rbind(as.numeric(scaled_new_country_data), center)) # Convert to numeric vector
      })
      
      # The predicted cluster is the one with the minimum distance
      predicted_cluster <- which.min(distances)
      
      # Map cluster number to priority label
      priority_map_for_prediction <- c("Low Priority", "Mid Priority", "High Priority") # Assuming 1=Low, 2=Mid, 3=High
      
      predicted_priority <- if (input$k_value == 3 && predicted_cluster %in% 1:3) {
         priority_map_for_prediction[predicted_cluster]
      } else {
         paste("Cluster", predicted_cluster)
      }
      
      output$prediction_output <- renderPrint({
         cat("Predicted Cluster:", predicted_cluster, "\n")
         cat("Predicted Priority:", predicted_priority, "\n")
      })
   })
}

# Run the application
shinyApp(ui = ui, server = server)
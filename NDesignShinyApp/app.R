library(shiny)
library(ggplot2)
library(dplyr)

# Load the dataframe
df <- read.csv("dfDesign.csv")

df$outcome <- ifelse(is.na(df$outcome), "Negative Affect", df$outcome)
df$outcome <- case_match(df$outcome,
           "ERLookDiff" ~ "Task-based Ratings",
           "neoN" ~ "Neuroticism",
           "neoN1" ~ "N1:Anxiety",
           "neoN2" ~ "N2:Hostility",
           "neoN3" ~ "N3:Depression",
           "neoN4" ~ "N4:Self-Consciousness",
           "neoN5" ~ "N5:Impulsiveness",
           "neoN6" ~ "N6:Vulnerability",
           "NEONother" ~ "N:Other",
           "NEONX" ~ "N:Other+Self",
           "PA" ~ "Positive Affect",
           "Negative Affect" ~ "Negative Affect",
           "BDI" ~ "BDI",
           "STAI" ~ "STAI")

df$data <- case_match(df$data,
                      "IAPS" ~ "scenes",
                      "PFA" ~ "faces")

df$contrast <- case_match(df$contrast,
                          "controlCond" ~ "Neutral Scenes/Shapes",
                          "implBaseline" ~ "Implicit Baseline")

df$trainsVsFull <- case_match(df$trainsVsFull,
                              "AHAB" ~ "AHAB2 subsample",
                              "full" ~ "Full Data",
                              "train" ~ "Training Data Only")

df$rescale <- case_match(df$rescale,
                         "cente" ~ "Image-wise centering",
                         "nocen" ~ "No image-wise scaling",
                         "zscor" ~ "Image-wise z-scoring")

df$algorithm <- case_match(df$algorithm,
                           "pcr" ~ "Principal Component Regression",
                           "pls" ~ "Partial Least Squares",
                           "rf" ~ "Random Forest",
                           "svr" ~ "Support Vector Regression")

names(df)[2] <- "task"
names(df)[4] <- "data"



# Define UI
ui <- fluidPage(
  titlePanel("Dynamic subsetting and plotting of design factor influencing cross-validated correlations"),
  fluidRow(
    column(4,
           h3("Filter Dataframe"),
           uiOutput("checkboxes_ui")
    ),
    column(8,
           h3("Plot Settings"),
           selectInput("xvar", "Select X-axis variable:", choices = names(df)[-7]),
           selectInput("colorvar", "Select Color variable:", choices = c("None", names(df)[-7])),
           selectInput("facetvar", "Select Facet variable:", choices = c("None", names(df)[-7])),
           h3("Boxplot"),
           plotOutput("boxplot")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  # Generate UI for checkboxes dynamically based on dataframe columns
  output$checkboxes_ui <- renderUI({
    checkbox_ui <- lapply(names(df)[-7], function(col) {
      unique_vals <- unique(df[[col]])
      checkboxGroupInput(inputId = col, 
                         label = paste("Select", col), 
                         choices = unique_vals, 
                         selected = unique_vals)
    })
    do.call(tagList, checkbox_ui)
  })
  
  # Reactive expression to subset dataframe based on checkbox inputs
  filtered_df <- reactive({
    sub_df <- df
    for(col in names(df)[-7]) {
      if(!is.null(input[[col]])) {
        sub_df <- sub_df[sub_df[[col]] %in% input[[col]], ]
      }
    }
    return(sub_df)
  })
  
  # Reactive expression to generate the ggplot boxplot
  output$boxplot <- renderPlot({
    req(input$xvar)
    p <- ggplot(filtered_df(), aes_string(x = input$xvar, y = "cvCorr")) +
      geom_boxplot() +
      geom_hline(yintercept = 0, color = "grey") +
      labs(y = "Correlation (cross-validated)", x = input$xvar) +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    
    if (input$colorvar != "None") {
      p <- p + aes_string(color = input$colorvar)
    }
    
    if (input$facetvar != "None") {
      p <- p + facet_wrap(as.formula(paste("~", input$facetvar)))
    }
    
    p
  })
}

# Run the application 
shinyApp(ui = ui, server = server)




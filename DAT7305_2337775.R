# Title: Customized Decision Support System in R for Predicting Patient-Specific Diabetes Risk Using Comprehensive Healthcare Data
# Author: Muhammad Salman
# Date: 21-08-2024

# Load necessary libraries
library(dplyr)
library(caret)
library(ROCR)
library(shiny)

# Extend the sample dataset to 150 values
extended_data <- data.frame(
  Pregnancies = rep(c(6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 4, 10, 10, 1, 5, 7, 0, 7, 1, 1, 3, 8, 7), length.out = 150),
  Glucose = rep(c(148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166, 100, 118, 107, 103, 115, 126, 99, 196), length.out = 150),
  BloodPressure = rep(c(72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 92, 74, 80, 60, 72, 0, 84, 74, 30, 70, 88, 84, 90), length.out = 150),
  SkinThickness = rep(c(35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 0, 0, 0, 23, 19, 0, 47, 0, 38, 30, 41, 0, 0), length.out = 150),
  Insulin = rep(c(0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 0, 0, 846, 175, 0, 230, 0, 83, 96, 235, 0, 0), length.out = 150),
  BMI = rep(c(33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0, 37.6, 38.0, 27.1, 30.1, 25.8, 30.0, 45.8, 29.6, 43.3, 34.6, 39.3, 35.4, 39.8), length.out = 150),
  DiabetesPedigreeFunction = rep(c(0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.191, 0.537, 1.441, 0.398, 0.587, 0.484, 0.551, 0.254, 0.183, 0.529, 0.704, 0.388, 0.451), length.out = 150),
  Age = rep(c(50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 30, 34, 57, 59, 51, 32, 31, 31, 33, 32, 27, 50, 41), length.out = 150),
  Outcome = rep(c(1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1), length.out = 150),
  FamilyHistory = rep(c(1, 0), length.out = 150) # Add FamilyHistory column
)

# Convert Outcome and FamilyHistory to factors
extended_data$Outcome <- as.factor(extended_data$Outcome)
extended_data$FamilyHistory <- as.factor(extended_data$FamilyHistory)

# Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(extended_data$Outcome, p = 0.7, list = FALSE)
trainData <- extended_data[trainIndex,]
testData <- extended_data[-trainIndex,]

# Train a logistic regression model with FamilyHistory included
model <- train(Outcome ~ Glucose + BloodPressure + BMI + Age + FamilyHistory, 
               data = trainData, method = "glm", family = "binomial")

# Make predictions on the test set
predictions <- predict(model, testData)

# Confusion matrix to evaluate the model
conf_matrix <- confusionMatrix(predictions, testData$Outcome)
print(conf_matrix)

# ROC curve and AUC calculation for the test data
pred_prob_test <- predict(model, testData, type = "prob")[,2]
pred_test <- prediction(pred_prob_test, testData$Outcome)
perf_test <- performance(pred_test, "tpr", "fpr")
plot(perf_test, colorize = TRUE)

# Calculate and print AUC for the test data
auc_test <- performance(pred_test, "auc")
auc_value_test <- auc_test@y.values[[1]]
print(paste("AUC:", auc_value_test))

# Sample Clinical Coding (ICD-10 Codes)
clinical_codes <- data.frame(
  Symptom = c("High Blood Sugar", "Obesity", "Family History of Diabetes"),
  ICD10 = c("E11", "E66", "Z83.3")
)

# Define UI for the app
ui <- fluidPage(
  titlePanel("Diabetes Risk Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Enter Your Medical History"),
      numericInput("glucose", "Glucose Level", value = 100, min = 0),
      numericInput("bp", "Blood Pressure", value = 70, min = 0),
      numericInput("bmi", "BMI", value = 25, min = 0),
      numericInput("age", "Age", value = 30, min = 0),
      checkboxInput("family_history", "Family History of Diabetes", value = FALSE),
      actionButton("predict_btn", "Predict Diabetes Risk")
    ),
    
    mainPanel(
      h3("Prediction Results"),
      verbatimTextOutput("prediction_text"),
      plotOutput("roc_plot"),
      verbatimTextOutput("recommendation_text"),
      verbatimTextOutput("clinical_codes_text")
    )
  )
)

# Define server logic
server <- function(input, output) {
  observeEvent(input$predict_btn, {
    
    # Convert user inputs to a data frame
    user_data <- data.frame(
      Glucose = input$glucose,
      BloodPressure = input$bp,
      BMI = input$bmi,
      Age = input$age,
      FamilyHistory = as.factor(ifelse(input$family_history, 1, 0)) # Convert to factor
    )
    
    # Make prediction for the user input
    pred_prob_user <- predict(model, user_data, type = "prob")[, 2]
    
    # Update the prediction text
    output$prediction_text <- renderText({
      paste("The predicted probability of having diabetes is:", round(pred_prob_user * 100, 2), "%")
    })
    
    # ROC curve plot (for the entire test dataset, not user input)
    output$roc_plot <- renderPlot({
      plot(perf_test, colorize = TRUE)
    })
    
    # Recommendations and Prescription
    output$recommendation_text <- renderText({
      if (pred_prob_user > 0.5) {
        "Recommendation: Please consult with your healthcare provider for further tests and lifestyle changes. Prescriptions may include Metformin or lifestyle interventions."
      } else {
        "Recommendation: Maintain a healthy lifestyle to reduce your risk of diabetes. Regular check-ups are recommended."
      }
    })
    
    # Clinical Coding Text
    output$clinical_codes_text <- renderText({
      if (input$family_history) {
        paste("Clinical Codes (ICD-10):", paste(clinical_codes$ICD10, collapse = ", "))
      } else {
        "No specific clinical codes relevant to the symptoms provided."
      }
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)

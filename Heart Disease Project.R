# Project: Detect Heart Disease With ML

library(randomForest)
library(caTools)
library(e1071)
library(pROC)

data <- read.csv('heart.csv') # importing real data
data$sex <- as.factor(data$sex) # gender
data$cp <- as.factor(data$cp)  # chest pain type
data$fbs <- as.factor(data$fbs)  # fasting blood sugar
data$restecg <- as.factor(data$restecg)  # resting electrocardiographic results
data$exang <- as.factor(data$exang)  # exercise-induced angina
data$slope <- as.factor(data$slope)  # slope of the ST segment
data$ca <- as.factor(data$ca)  # number of major vessels by fluoroscopy
data$thal <- as.factor(data$thal)  # thalassemia

sum(is.na(data)) # check missing values

set.seed(5) # establish reproducibility
split <- sample.split(data$target, SplitRatio = 0.7) # split data into 70% training, 30% testing
train_data <- subset(data, split == TRUE) # takes the 70% from the split
test_data <- subset(data, split == FALSE) # takes the 30% left from the split

# Logistic Regression Model

logistic_model <- glm(target ~ ., data = train_data, family = binomial) # fit the model

summary(logistic_model) # summary of the model

logistic_predictions_prob <- predict(logistic_model, test_data, type = "response") # make predictions on the test set

logistic_predictions <- ifelse(logistic_predictions_prob > 0.5, 1, 0) # match probabilities to binary outcomes

logistic_cm <- table(test_data$target, logistic_predictions)
print(logistic_cm) # evaluate model using confusion matrix

logistic_accuracy <- sum(diag(logistic_cm)) / sum(logistic_cm)
cat("Logistic Regression Accuracy:", logistic_accuracy) # evaluate model using accuracy

# ROC curve and AUC (want AUC as close to 1 as possible)
roc_log <- roc(test_data$target, logistic_predictions_prob)
plot(roc_log)
cat("Logistic Regression AUC:", auc(roc_log))

logistic_predictions <- ifelse(logistic_predictions_prob > 0.25, 1, 0) # tuning probabilities to binary outcomes so test is safer

logistic_cm <- table(test_data$target, logistic_predictions)
print(logistic_cm) # evaluate tuned model using confusion matrix

logistic_accuracy <- sum(diag(logistic_cm)) / sum(logistic_cm)
cat("Logistic Regression Accuracy:", logistic_accuracy) # evaluate tuned model using accuracy

# ROC curve and AUC (want AUC as close to 1 as possible)
roc_log <- roc(test_data$target, logistic_predictions_prob)
plot(roc_log)
cat("Logistic Regression AUC:", auc(roc_log))

# Random Forest Model
train_data$target <- as.factor(train_data$target)
test_data$target <- as.factor(test_data$target)

rf_model <- randomForest(target ~ ., data = train_data, ntree = 100)

print(rf_model) # model Summary

rf_predictions <- predict(rf_model, test_data)

rf_cm <- table(test_data$target, rf_predictions)
print(rf_cm) # confusion matrix

rf_accuracy <- sum(diag(rf_cm)) / sum(rf_cm)
cat("Random Forest Accuracy:", rf_accuracy) # accuracy

rf_prob <- predict(rf_model, test_data, type = "prob")[,2]
roc_rf <- roc(test_data$target, rf_prob)
plot(roc_rf)
cat("Random Forest AUC:", auc(roc_rf))

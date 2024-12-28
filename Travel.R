
# Objective:
# To develop a model for predicting which client might buy the recently offered travel package.

library(dplyr)
library(ggplot2)
library(caret)
library(randomForest)
library(fastDummies)
library(kernlab)
library(rpart)
library(rpart.plot)

set.seed(777)
data <- read.csv("Tourism.csv")
data <- na.omit(data) # dropping rows with missing values.

str(data) #checking the data types of columns

unique_counts <- data %>%
  summarise_all(n_distinct)

# Gender has 3 unique values. Looks like there is misspell of "female" to "fe male"
# customer id is not needed as every customer has unique value

data <- data %>%
  mutate(Gender = ifelse(Gender == "Fe Male", "Female", Gender)) # this changes any misspell of "female"


data <- data %>%
  select(-CustomerID) #removing customer id

# List of categorical variables
catagory_col <- c('TypeofContact', 'CityTier', 'Occupation', 'Gender', 'OwnCar',
             'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
             'MaritalStatus', 'PitchSatisfactionScore','NumberOfPersonVisiting', 
             'Passport','NumberOfChildrenVisiting', 'Designation')

#Converting each categorical variable to factor
data <- data %>%
  mutate(across(all_of(catagory_col), as.factor))


## Data preparation for modeling

X <- select(data, -ProdTaken)  # Removing the target variable
Y <- data$ProdTaken            # Target variable

# removing unnecessary columns
X <- select(X, -c(DurationOfPitch, NumberOfFollowups, ProductPitched, PitchSatisfactionScore))

# Splitting the data into training and test set
trainIndex <- createDataPartition(Y, p = 0.70, list = FALSE, times = 1)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- Y[trainIndex]
y_test <- Y[-trainIndex]

# Converting some columns from categorical to numeric in both training and testing sets
columns_to_convert <- c('NumberOfPersonVisiting', 'Passport', 'OwnCar')
for(column in columns_to_convert) {
  X_train[[column]] <- as.numeric(as.character(X_train[[column]]))
  X_test[[column]] <- as.numeric(as.character(X_test[[column]]))
}

# creating dummy variables 
dummy_columns <- c('TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'Designation', 'CityTier')

X_train <- dummy_cols(X_train, select_columns = dummy_columns, remove_first_dummy = TRUE)
X_test <- dummy_cols(X_test, select_columns = dummy_columns, remove_first_dummy = TRUE)


## Scaling the features
preProcValues <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProcValues, X_train)
X_test_scaled <- predict(preProcValues, X_test)

## SVM model training and evaluation using scaled data
y_train <- as.factor(y_train)
y_train <- factor(y_train, levels = c(1, 0))
y_test <- factor(y_test, levels = c(1, 0))
svm_model <- ksvm(y_train ~ ., data = X_train_scaled, type = "C-svc", kernel = "vanilladot")
predictions <- predict(svm_model, X_test_scaled)  # making predictions 
y_test <- as.factor(y_test)
conf_matrix <- confusionMatrix(predictions, y_test) # creating confusion matrix
print(conf_matrix)

# improving the model with rbfdot
svm_model <- ksvm(y_train ~ ., data = X_train_scaled, type = "C-svc", kernel = "rbfdot")
predictions <- predict(svm_model, X_test_scaled) # making predictions
y_test <- as.factor(y_test)
conf_matrix <- confusionMatrix(predictions, y_test) # creating confusion matrix
print(conf_matrix)


# The initial SVM model with the vanilladot kernel settings achieved a sensitivity 
# of 18.06% and a specificity of 98.42%. This model is extremely conservative, 
# significantly minimizing false positives but at the cost of missing a large number 
# of true positives, as reflected in its low sensitivity. In contrast, the improved 
# SVM model using the rbfdot kernel exhibits better performance with a sensitivity 
# of 26.87% and a specificity of 98.22%. Although still conservative, this model 
# captures a greater proportion of true buyers with its increase in sensitivity, 
# while maintaining a high specificity, which minimizes targeting non-buyers. 
# The balanced accuracy also improved from 58.24% to 62.55%, indicating a more 
# effective balance between identifying true buyers and avoiding false alarms. 
# Overall, the rbfdot kernel model not only improves the accuracy and reduces 
# false negatives but also maintains high specificity, making it a more effective model.


## Decision Tree

# Training the decision tree
tree_model <- rpart(y_train ~ ., data = X_train_scaled, method = "class")

# Predicting and evaluating
predictions <- predict(tree_model, X_test_scaled, type = "class") # making predictions
y_test <- as.factor(y_test)
conf_matrix <- confusionMatrix(predictions, y_test)  # creating confusion matrix
print(conf_matrix)

rpart.plot(tree_model, extra = 104)

# The Decision Tree model achieves a sensitivity of 31.28% and a specificity of 
# 95.94%, showing it has a reasonable capability to identify true buyers (class 1) 
# while maintaining a high rate of correctly identifying non-buyers (class 0). 
# In contrast, the SVM model with the rbfdot kernel offers a slightly lower sensitivity 
# at 26.87% but an almost similar specificity at 98.22%. Although the SVM model is 
# slightly more conservative in predicting non-buyers correctly, the Decision Tree 
# provides a better sensitivity, meaning it is less likely to miss buyers. 
# However, both models still show a preference for specificity over sensitivity, though 
# the Decision Tree edges out with a better balance between the two, because its slightly 
# higher balanced accuracy of 63.61% compared to the SVM's 62.55%. This makes the 
# Decision Tree potentially more favorable for scenarios where missing out on actual 
# buyers can be costly, even though it sacrifices some accuracy in identifying non-buyers.



#It takes about 5 minutes to run all the code

#install.packages("dfms")
#install.packages("forecast")
#install.packages("randomForest")

#Contents
#PART1|ARIMA                        —— Lines 16-39
#PART2|PCA+DMF                      —— Lines 44-67
#PART3|PCA+ARIMA                    —— Lines 72-184
#PART4|Random Forest for PPI        —— Lines 189-327
#PART4|Random Forest for CPI        —— Lines 332-468
#PART5|Inflation Rate Prediction    —— Lines 473-505




#PART1|ARIMA################################################################################################
rm(list = ls())
load(url("https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"))
library(forecast)
cpi_ts <- ts(cpi[, 2])
arima_model <- auto.arima(cpi_ts)
predictions <- forecast(arima_model, h = 30)
# Plotting the historical CPI data
plot(cpi_ts, main = "CPI Prediction", ylab = "CPI Level")
# Add the predicted values to the plot
lines(predictions$mean, col = "red")  # Mean predictions
lines(predictions$lower[, "95%"], col = "blue", lty = "dashed")  # 95% confidence interval lower bound
lines(predictions$upper[, "95%"], col = "blue", lty = "dashed")  # 95% confidence interval upper bound
ppi_ts <- ts(ppi[, 2])
arima_model <- auto.arima(ppi_ts)
predictions <- forecast(arima_model, h = 30)
# Plotting the historical PPI data
plot(ppi_ts, main = "PPI Prediction", ylab = "PPI Level")
# Add the predicted values to the plot
lines(predictions$mean, col = "red")  # Mean predictions
lines(predictions$lower[, "95%"], col = "blue", lty = "dashed")  # 95% confidence interval lower bound
lines(predictions$upper[, "95%"], col = "blue", lty = "dashed")  # 95% confidence interval upper bound
#结论：此次是小组的初步探索，使用auto.arima通过对PPI和CPI的1-168期的数据进行简单预测，未引入X及fake.X变量
############################################################################################################




#PART2|PCA+DMF##############################################################################################
rm(list = ls())
load(url("https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"))
library(dfms)
library(forecast)
selected_cols <- c(14:18, 26, 28, 30, 34:35, 41, 48:51, 55, 57, 64, 66:70, 75:81, 94, 100:125, 127:129, 140:142, 147, 150)
cpi <- as.matrix(ts(cpi[,2], start = 1, frequency = 12)) # Assuming monthly data
X <- as.matrix(ts(X[,selected_cols], start = 1, frequency = 12))
fake.testing.X <- as.matrix(ts(fake.testing.X[,selected_cols], start = 169, frequency = 12))
pca <- prcomp(X, scale = TRUE)
PC <- pca$x
eig_ratio <- pca$sdev[-1]^2 / pca$sdev[-length(pca$sdev)]^2
plot(eig_ratio, type = "b", ylab = "Eigenvalue Ratio", xlab = "Component Number")
n_factors <- sum(pca$sdev^2 > 1)
n_factors <- 19
factors <- pca$x[, 1:n_factors]
dfm_data <- data.frame(cbind(cpi, factors))
# Fit the DFM model
dfm_model <- DFM(dfm_data, r= 1, p = 3) 
plot(dfm_model, method = "all", type = "individual")
forecast_values <- predict(dfm_model, newdata = fake.testing.X, h=30)
plot(forecast_values)
#结论：dfm()模型未能有效区分因变量和自变量，所做未来30期的预测无实际意义，没有参考实际值。团队考虑后舍弃,相关代码注释见后续模型
############################################################################################################




#PART3|PCA+ARIMA##############################################################################################
rm(list = ls())
load(url("https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"))
library(ggplot2)
library(forecast)
#选择指定的 CPI 变量列，并进行矩阵化
selected_cols_cpi <- c(14:18, 26, 28, 30, 34:35, 41, 48:51, 55, 57, 64, 66:70, 75:81, 94, 100:125, 127:129, 140:142, 147, 150)
selected_cols_ppi <- c(3:13, 18, 26, 34:35, 37, 44, 47, 49, 51, 64, 71:79, 82:93, 102:108, 129, 142, 148:149)
cpi <- as.matrix(ts(cpi[,2], start = 1, frequency = 12)[, "CPI"]) # Assuming monthly data
ppi <- as.matrix(ts(ppi[,2], start = 1, frequency = 12)[, "PPI"]) # Assuming monthly data
X_cpi <- as.matrix(ts(X[,selected_cols_cpi], start = 1, frequency = 12))
X_ppi <- as.matrix(ts(X[,selected_cols_ppi], start = 1, frequency = 12))
fake.testing.X_cpi <- as.matrix(ts(fake.testing.X[,selected_cols_cpi], start = 169, frequency = 12))
fake.testing.X_ppi <- as.matrix(ts(fake.testing.X[,selected_cols_ppi], start = 169, frequency = 12))

#Perform PCA on X and fake.testing.X
pca_cpi <- prcomp(X_cpi[,-1])
pca_test_cpi <- prcomp(fake.testing.X_cpi[,-1])
pca_ppi <- prcomp(X_ppi[,-1])
pca_test_ppi <- prcomp(fake.testing.X_ppi[,-1])

# Extract the principal components_cpi
##方法1：Elbow method
eig_ratio_cpi <- pca_cpi$sdev[-1]^2 / pca_cpi$sdev[-length(pca_cpi$sdev)]^2
plot(eig_ratio_cpi, type = "b", ylab = "Eigenvalue Ratio_cpi", xlab = "Component Number")
##波动不明显，不能选择具体主成分，因此选择方法2
##方法2：Kaiser criterion，得到主成分因子n_factors为19
n_factors_cpi <- sum(pca_cpi$sdev^2 > 1)
##选择主成分，其中对于fake.testing.X得到的pca_test来说，使用相同主成分保持后续regressors一致。
##此外，使用方法2对于pca_test来说得到的主成分因子为20，差别不大，因此直接选定。
factors_cpi <- pca_cpi$x[, 1:n_factors_cpi]
factors_test_cpi <- pca_test_cpi$x[, 1:n_factors_cpi]
##同时，当选择方法1时，由于testing.X和X的巨大差异，不能保证主成分因子相同，因此用到下行代码，但团队评估这种预测结果产生较大偏差，因此弃用
##factors_test_cpi <- predict(pca_cpi, newdata = fake.testing.X_cpi[,-1])[,1:n_factors_cpi]

#Extract the principal components_ppi,同理如下，得到主成分因子n_factors_ppi为28，
eig_ratio_ppi <- pca_ppi$sdev[-1]^2 / pca_ppi$sdev[-length(pca_ppi$sdev)]^2
plot(eig_ratio_ppi, type = "b", ylab = "Eigenvalue Ratio_ppi", xlab = "Component Number")
n_factors_ppi <- sum(pca_ppi$sdev^2 > 1)
factors_ppi <- pca_ppi$x[, 1:n_factors_ppi]
factors_test_ppi <- pca_test_ppi$x[, 1:n_factors_ppi]

# Estimate an auto-regressive model with external regressors (the factors)-cpi
model_cpi <- auto.arima(cpi, xreg = factors_cpi)
forecastcpi <- forecast(model_cpi, xreg = factors_cpi)
#观察模型效果，用MSE
forecastcpi_mse <- mean((forecastcpi$mean - cpi)^2)
forecastcpi_rmse <- sqrt(forecastcpi_mse)
print(paste0("主成分分析下的ARIAMA模型得到有关CPI的MSE: ", forecastcpi_mse))
print(paste0("主成分分析下的ARIAMA模型得到有关CPI的RMSE: ", forecastcpi_rmse))
#MSE为0.0585115857973596, RMSE为0.24189168195157
#作图看效果
plot_cpidata <- data.frame(
  Month = 1:168,
  Actual_CPI = cpi,
  Forecasted_CPI = forecastcpi$mean
)
ggplot(plot_cpidata, aes(x = Month)) +
  geom_line(aes(y = Actual_CPI), colour = "blue") +
  geom_line(aes(y = Forecasted_CPI), colour = "red") +
  labs(x = "Month", y = "CPI", 
       title = "Actual vs Forecasted CPI",
       subtitle = "Blue: Actual CPI, Red: Forecasted CPI") +
  theme_minimal()

# Estimate an auto-regressive model with external regressors (the factors)-ppi
model_ppi <- auto.arima(ppi, xreg = factors_ppi)
forecastppi <- forecast(model_ppi, xreg = factors_ppi)
#观察模型效果，用MSE
forecastppi_mse <- mean((forecastppi$mean - ppi)^2)
forecastppi_rmse <- sqrt(forecastppi_mse)
print(paste0("主成分分析下的ARIAMA模型得到有关PPI的MSE: ", forecastppi_mse))
print(paste0("主成分分析下的ARIAMA模型得到有关PPI的RMSE: ", forecastppi_rmse))
#MSE为0.112467107193535, RMSE为0.335361159339502
#作图看效果
plot_ppidata <- data.frame(
  Month = 1:168,
  Actual_PPI = ppi,
  Forecasted_PPI = forecastppi$mean
)
ggplot(plot_ppidata, aes(x = Month)) +
  geom_line(aes(y = Actual_PPI), colour = "blue") +
  geom_line(aes(y = Forecasted_PPI), colour = "red") +
  labs(x = "Month", y = "PPI", 
       title = "Actual vs Forecasted PPI",
       subtitle = "Blue: Actual PPI, Red: Forecasted PPI") +
  theme_minimal()

#预测fakecpi
forecast_testcpi <- forecast(model_cpi, xreg = factors_test_cpi)
# Create a data frame for plotting
plot_data_cpi <- data.frame(Month = c(1:198), CPI = c(cpi, forecast_testcpi$mean))
# Plot the data
ggplot(plot_data_cpi, aes(x = Month, y = CPI)) +
  geom_line() +
  geom_vline(xintercept = 168, linetype = "dashed")  # to show the point where out-of-sample forecasting starts

#预测fakeppi
forecast_testppi <- forecast(model_ppi, xreg = factors_test_ppi)
# Create a data frame for plotting
plot_data_ppi <- data.frame(Month = c(1:198), PPI = c(ppi, forecast_testppi$mean))
# Plot the data
ggplot(plot_data_ppi, aes(x = Month, y = PPI)) +
  geom_line() +
  geom_vline(xintercept = 168, linetype = "dashed")  # to show the point where out-of-sample forecasting starts

# 将预测值转换为数据框
forecast_cpi <- data.frame(Month = 169:198, Forecasted_CPI = forecast_testcpi$mean)
forecast_ppi <- data.frame(Month = 169:198, Forecasted_PPI = forecast_testppi$mean)
save(forecast_cpi, file = "forecast_cpi.RData")
save(forecast_ppi, file = "forecast_ppi.RData")
#结论：pca+arima模型较好预测cpi与ppi，团队较好的预测了cpi和ppi各未来30期的值，并且1-168期的值在保证不过拟合的情况下进行
############################################################################################################




#PART4|Random Forest for PPI##########################################################################################
rm(list = ls())
# Load the dataset
load(url("https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"))
library(randomForest)
library(ggplot2)

# Selecting the specified PPI variable column
selected_cols <- c(3:13, 18, 26, 34:35, 37, 44, 47, 49, 51, 64, 71:79, 82:93, 102:108, 129, 142, 148:149)
X <- X[, selected_cols]
fake.testing.X <- fake.testing.X[, selected_cols]

# Splitting the data into training and test sets
# Using the first 120 observations as training data (train) and the remaining 48 observations as test data (test)
set.seed(123)
train_indices <- sample(1:168, 120, replace = FALSE)
train_ppi <- ppi[train_indices, 2]

train_X <- X[train_indices, -1]
test_ppi <- ppi[-train_indices, 2]
test_X <- X[-train_indices, -1]
train_ppi <- data.frame(train_ppi)
train_ppi <- as.numeric(train_ppi[[1]])


#------------ Training the Random Forest model


# Train a random forest regression model using the randomForest() function.
# Adjust the hyperparameters ntree and mtry.
# "ntree" represents the number of trees in the forest. A higher number of trees generally leads to better model performance but also requires more computational time.
# "mtry" represents the number of randomly selected features to consider at each node for splitting. A smaller mtry value can lead to overfitting, while a larger value can lead to underfitting.

# Set the range of values for ntree and mtry.
ntree_values <- seq(400, 600, by = 50)
mtry_values <- seq(20, 40, by = 5)

# Initialize a result matrix.
results <- matrix(NA, nrow = length(ntree_values), ncol = length(mtry_values))

# Iterate through different combinations of ntree and mtry values.
for (i in 1:length(ntree_values)) {
  for (j in 1:length(mtry_values)) {
    # Train the random forest model.
    set.seed(123)
    rf <- randomForest(train_X, train_ppi, ntree = ntree_values[i], mtry = mtry_values[j])
    # Perform cross-validation and calculate the mean error
    set.seed(123)
    cv_error <- rep(NA, 10)
    for (k in 1:10) {
      cv_indices <- sample(1:120, 12, replace = FALSE)
      cv_train_X <- train_X[-cv_indices, ]
      cv_train_ppi <- train_ppi[-cv_indices]
      cv_test_X <- train_X[cv_indices, ]
      cv_test_ppi <- train_ppi[cv_indices]
      cv_rf <- randomForest(cv_train_X, cv_train_ppi, ntree = ntree_values[i], mtry = mtry_values[j])
      cv_error[k] <- mean((predict(cv_rf, cv_test_X) - cv_test_ppi)^2)
    }
    results[i, j] <- mean(cv_error)
  }
# This step may take a while as it computes the mean error for all combinations and selects the combination with the smallest mean error.
}

# Find the ntree and mtry values with the lowest mean error.
min_error <- min(results)
min_indices <- which(results == min_error, arr.ind = TRUE)
optimal_ntree <- ntree_values[min_indices[1]]
optimal_mtry <- mtry_values[min_indices[2]]

# Train a random forest regression model using the randomForest() function with 500 trees and 30 random variables considered for splitting at each node.
rf_model <- randomForest(train_X, y = train_ppi, ntree = 500, mtry = 30)

# Use the predict() function to make predictions on the test data.
rf_pred <- predict(rf_model, newdata=test_X)
print(cbind(test_ppi, rf_pred))

# Combine test_ppi and rf_pred into a data frame.
df <- data.frame(month = 1:48, 
                 observed = test_ppi, 
                 predicted = rf_pred)
# Use ggplot to create a line plot.
# Plot the actual test data (48 observations in test) in black and the predicted data from random forest in red.
ggplot(df, aes(x = month)) + 
  geom_line(aes(y = PPI, color = "observed_ppi")) + 
  geom_line(aes(y = predicted, color = "predicted_ppi")) + 
  labs(title = "Observed and Predicted PPI", y = "PPI") +
  scale_color_manual(values = c("black", "red"), 
                     name = "PPI",
                     labels = c("Observed PPI", "Predicted PPI"))

#——————Evaluate the training performance of the random forest model:
# Convert the variable test_ppi of type data.frame to numeric type.
test_ppi <- as.numeric(test_ppi$PPI)
# Calculate the Mean Squared Error (MSE) of the model.
rf_mse <- mean((rf_pred - test_ppi)^2)
# Print the MSE. A smaller MSE indicates better prediction performance of the model.
print(paste0("Random forest MSE: ", rf_mse))
# MSE = 0.0373501949785841
# Meaning that the random forest model's predicted values have a smaller difference from the actual values.

# Calculate the Root Mean Squared Error (RMSE) of the model.
rf_rmse <- sqrt(rf_mse)
# Print the RMSE. A smaller RMSE indicates better prediction performance of the model.
print(paste0("Random forest RMSE: ", rf_rmse))
# RMSE = 0.193261985342654
# Meaning that the random forest model's predicted values have a smaller difference from the actual values.

#——————Generate the final predictions using the trained random forest model with fake.testing.X:

# Split the data into training set and test set.
# Use all observations as training data.
set.seed(123)
train_ppi2 <- ppi[, 2]
train_X2 <- X[, -1]
train_ppi2 <- data.frame(train_ppi2)
train_ppi2 <- as.numeric(train_ppi2[[1]])

# Train a random forest regression model using the randomForest() function with 500 trees and 30 random variables considered for splitting at each node.
rf_model2 <- randomForest(train_X2, y = train_ppi2, ntree = 500, mtry = 30)
# Use the trained model to predict fake.testing.X.
fake_testing_pred <- predict(rf_model2, newdata = fake.testing.X)
# Merge the predicted values with the original data.
predicted_ppi <- c(train_ppi2, fake_testing_pred)
print (predicted_ppi)

# Original data set: observed_ppi
observed_ppi <- data.frame(month = 1:168, PPI = train_ppi2)
# Predicted data set: forecast_ppi
forecast_ppi <- data.frame(month = 169:198, PPI = fake_testing_pred)
# Merge the data sets: observed_and_forecast_ppi
observed_and_forecast_ppi <- rbind(observed_ppi, forecast_ppi)
# Add a color variable.
observed_and_forecast_ppi$color <- ifelse(observed_and_forecast_ppi$month <= 168, "observed_ppi", "predicted_ppi")
# Plot the graph.
ggplot(observed_and_forecast_ppi, aes(x = month, y = PPI, color = color)) +
  geom_line() +
  labs(title = "Observed and Predicted PPI", y = "PPI") +
  scale_color_manual(name = "PPI", values = c("observed_ppi" = "black", "predicted_ppi" = "red"))
############################################################################################################




#PART4|Random Forest for CPI##########################################################################################
rm(list = ls())
# Load the dataset
load(url("https://github.com/zhentaoshi/Econ5821/raw/main/data_example/dataset_inf.Rdata"))
library(randomForest)
library(ggplot2)

# Selecting the specified CPI variable column
selected_cols <- c(14:18, 26, 28, 30, 34:35, 41, 48:51, 55, 57, 64, 66:70, 75:81, 94, 100:125, 127:129, 140:142, 147, 150)
X <- X[, selected_cols]
fake.testing.X <- fake.testing.X[, selected_cols]

# Splitting the data into training and test sets
# Using the first 120 observations as training data (train) and the remaining 48 observations as test data (test)
set.seed(123)
train_indices <- sample(1:168, 120, replace = FALSE)
train_cpi <- cpi[train_indices, 2]

train_X <- X[train_indices, -1]
test_cpi <- cpi[-train_indices, 2]
test_X <- X[-train_indices, -1]
train_cpi <- data.frame(train_cpi)
train_cpi <- as.numeric(train_cpi[[1]])

#——————Training the Random Forest model

# Train a random forest regression model using the randomForest() function.
# Adjust the hyperparameters ntree and mtry.
# "ntree" represents the number of trees in the forest. A higher number of trees generally leads to better model performance but also requires more computational time.
# "mtry" represents the number of randomly selected features to consider at each node for splitting. A smaller mtry value can lead to overfitting, while a larger value can lead to underfitting.

# Set the range of values for ntree and mtry.
ntree_values <- seq(200, 400, by = 50)
mtry_values <- seq(40, 60, by = 5)

# Initialize a result matrix.
results <- matrix(NA, nrow = length(ntree_values), ncol = length(mtry_values))

# Iterate through different combinations of ntree and mtry values.
for (i in 1:length(ntree_values)) {
  for (j in 1:length(mtry_values)) {
    # Train the random forest model.
    set.seed(123)
    rf <- randomForest(train_X, train_cpi, ntree = ntree_values[i], mtry = mtry_values[j])
    # Perform cross-validation and calculate the mean error.
    set.seed(123)
    cv_error <- rep(NA, 10)
    for (k in 1:10) {
      cv_indices <- sample(1:120, 12, replace = FALSE)
      cv_train_X <- train_X[-cv_indices, ]
      cv_train_cpi <- train_cpi[-cv_indices]
      cv_test_X <- train_X[cv_indices, ]
      cv_test_cpi <- train_cpi[cv_indices]
      cv_rf <- randomForest(cv_train_X, cv_train_cpi, ntree = ntree_values[i], mtry = mtry_values[j])
      cv_error[k] <- mean((predict(cv_rf, cv_test_X) - cv_test_cpi)^2)
    }
    results[i, j] <- mean(cv_error)
  }
# This step may take a while as it computes the mean error for all combinations and selects the combination with the smallest mean error.
}

# Find the ntree and mtry values with the lowest mean error.
min_error <- min(results)
min_indices <- which(results == min_error, arr.ind = TRUE)
optimal_ntree <- ntree_values[min_indices[1]]
optimal_mtry <- mtry_values[min_indices[2]]

# Train a random forest regression model using the randomForest() function with 300 trees and 50 random variables considered for splitting at each node.
rf_model <- randomForest(train_X, y = train_cpi, ntree = 300, mtry = 50)

# Use the predict() function to make predictions on the test data.
rf_pred <- predict(rf_model, newdata=test_X)
print(cbind(test_cpi, rf_pred))

# Combine test_cpi and rf_pred into a data frame.
df <- data.frame(month = 1:48, 
                 observed = test_cpi, 
                 predicted = rf_pred)
# Use ggplot to create a line plot.
# Plot the actual test data (48 observations in test) in black and the predicted data from random forest in red.
ggplot(df, aes(x = month)) + 
  geom_line(aes(y = CPI, color = "observed_cpi")) + 
  geom_line(aes(y = predicted, color = "predicted_cpi")) + 
  labs(title = "Observed and Predicted CPI", y = "CPI") +
  scale_color_manual(values = c("black", "red"), 
                     name = "CPI",
                     labels = c("Observed CPI", "Predicted CPI"))

#——————Evaluate the training performance of the random forest model:
# Convert the variable test_cpi of type data.frame to numeric type.
test_cpi <- as.numeric(test_cpi$CPI)
# Calculate the Mean Squared Error (MSE) of the model.
rf_mse <- mean((rf_pred - test_cpi)^2)
# Print the MSE. A smaller MSE indicates better prediction performance of the model.
print(paste0("Random forest MSE: ", rf_mse))
# MSE = 0.0148394993208728
# Print the MSE. A smaller MSE indicates better prediction performance of the model.

# Calculate the Root Mean Squared Error (RMSE) of the model.
rf_rmse <- sqrt(rf_mse)
# Print the RMSE. A smaller RMSE indicates better prediction performance of the model.
print(paste0("Random forest RMSE: ", rf_rmse))
# RMSE = 0.121817483642016
# Meaning that the random forest model's predicted values have a smaller difference from the actual values.

#——————Generate the final predictions using the trained random forest model with fake.testing.X:

# Split the data into training set and test set.
# Use all observations as training data.
set.seed(123)
train_cpi2 <- cpi[, 2]
train_X2 <- X[, -1]
train_cpi2 <- data.frame(train_cpi2)
train_cpi2 <- as.numeric(train_cpi2[[1]])

# Train a random forest regression model using the randomForest() function with 300 trees and 50 random variables considered for splitting at each node.
rf_model2 <- randomForest(train_X2, y = train_cpi2, ntree = 300, mtry = 50)
# Use the trained model to predict fake.testing.X.
fake_testing_pred <- predict(rf_model2, newdata = fake.testing.X)
# Merge the predicted values with the original data.
predicted_cpi <- c(train_cpi2, fake_testing_pred)
print (predicted_cpi)

# Original data set: observed_cpi
observed_cpi <- data.frame(month = 1:168, CPI = train_cpi2)
# Predicted data set: forecast_cpi
forecast_cpi <- data.frame(month = 169:198, CPI = fake_testing_pred)
# Merge the data sets: observed_and_forecast_cpi
observed_and_forecast_cpi <- rbind(observed_cpi, forecast_cpi)
# Add a color variable.
observed_and_forecast_cpi$color <- ifelse(observed_and_forecast_cpi$month <= 168, "observed_cpi", "predicted_cpi")
# Plot the graph.
ggplot(observed_and_forecast_cpi, aes(x = month, y = CPI, color = color)) +
  geom_line() +
  labs(title = "Observed and Predicted CPI", y = "CPI") +
  scale_color_manual(name = "CPI", values = c("observed_cpi" = "black", "predicted_cpi" = "red"))
############################################################################################################




###PART5|Inflation Rate Prediction############################################################################

#In this part, we will use Random Forest to derive the prediction results of inflation rates based on CPI.

# According to the given formula indicating the relationship with inflation rate and CPI, we extract two parts of CPI values whose lengths are both 30
predicted_cpi_values1 <- predicted_cpi[157:186]
predicted_cpi_values2 <- predicted_cpi[169:198]

# We derive the predicted inflation rate according to the formula using the two parts of CPI values.
# And we obtain the results of inflation rate prediction. ( from month 169 to month 198 ）【THE RESULTS OF OUR PROJECT】
inflation_rate_predictions <- log(predicted_cpi_values2) - log(predicted_cpi_values1)
print(inflation_rate_predictions)

# Then we generate the historical inflation rate. ( from month 13 to month 168 ) 
inflation_rate_13to168 <- log(predicted_cpi[13:168]) - log(predicted_cpi[1:156])
month_inflation_rate_13to168 <- 13:168
historical_inflation_rate <- data.frame(month_inflation_rate_13to168, inflation_rate_13to168)
print(historical_inflation_rate)

# To generate plots, firstly we generate the observed inflation rates.     
observed_inflation_rate <- data.frame(month = 13:168, Inflation_rate = inflation_rate_13to168)
# Secondly we generate the predicted inflation rates.
predicted_inflation_rate <- data.frame(month = 169:198, Inflation_rate = inflation_rate_predictions)
# Thirdly we combine these two data frames into one.
all_inflation_rate <- rbind(observed_inflation_rate, predicted_inflation_rate)
# Next we add a color variable.
all_inflation_rate$color <- ifelse(all_inflation_rate$month <= 168, "observed_inflation_rate", "predicted_inflation_rate")
# Lastly we generate the plot of inflation rate prediction based on CPI(random forest).
ggplot(all_inflation_rate, aes(x = month, y = Inflation_rate, color = color)) +
  geom_line() +
  labs(title = "Observed and Predicted Inflation Rate", y = "Inflation Rate") +
  scale_color_manual(name = "Inflation Rate", values = c("observed_inflation_rate" = "black", "predicted_inflation_rate" = "red"))
############################################################################################################

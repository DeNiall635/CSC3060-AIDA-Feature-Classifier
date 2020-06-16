library(ggplot2)
library(gmodels)
library(plyr)

set.seed(3060)
#Load Feature Data
filepath <- paste(getwd(),"/40173800_features.csv", sep = "")
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "vertical3tile", "horizontal3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "unused_line")

feature_data <- read.csv(filepath, header = TRUE, col.names = feature_col_names)
feature_data$label <- tolower(as.factor(feature_data$label))
feature_data$classification <- NA

#Insert classification of Object (Living = 1 or Nonliving = 0)

living <- c("banana", "cherry", "flower", "pear")
nonliving <- c("envelope", "golfclub", "pencil", "wineglass")

define_object <- function()
{
  classifications <- c()
  for (i in 1:nrow(feature_data))
  {
    classification <- -1
    name <- feature_data[i,1]
    if (name %in% living)
    {
      classification <- 1
    }
    else if (name %in% nonliving)
    {
      classification <- 0
    }
    else
    {
      print("Undefined")
      return()
    }
    classifications <- c(classifications, classification)
  }
  return (classifications)
}

feature_data$classification <- define_object()


#**************************************************************************************************************************************************#

#1.1 Generate a Logistic Regression Model using glm function with the Training Data

#Create a training Dataset
shuffled_set <- feature_data[sample(nrow(feature_data)),]
training_data <- shuffled_set[1:132,]
test_data <- shuffled_set[133:160,]

#Plot the histogram for the verticalness feature
vert_hist <- ggplot(training_data, aes(x=verticalness, fill = as.factor(classification))) +
  geom_histogram(binwidth = .2, alpha = .5, position = 'identity')
vert_hist

ggsave('hist_verticalness.png', scale = 1, dpi = 400)

#Build the model
model_1_1 <- glm(classification ~ verticalness,
                 data = training_data,
                 family = 'binomial')
summary(model_1_1)

#Plot the fitted curve
x_range <- range(training_data$verticalness)
x_axis <- seq(x_range[1], x_range[2], length.out = 1000)

fitted_curve <- data.frame(verticalness = x_axis)
fitted_curve$classification <- predict(model_1_1, fitted_curve, type = "response")

curve_plot <- ggplot(training_data, aes(x=verticalness, y=classification)) + 
  geom_point(aes(colour = factor(classification)), 
             show.legend = T, position="dodge")+
  geom_line(data=fitted_curve, colour="orange", size=1)
curve_plot

ggsave('1.1_fitted_curve.png', scale = 1, dpi = 400)

#**************************************************************************************************************************************************#

#1.2 Calculate Accuracy of Model using the 160 items in the Feature Data

#Dataframe to store the accuracies for each p cut-off value
p_acc_table_1_2 <- data.frame(p = character(), accuracy = character())


for (p in seq(0.01,0.99, by = 0.01))
{
  test_data$predicted_val = predict(model_1_1, test_data, type="response")
  test_data$predicted_class = 0
  test_data$predicted_class[test_data$predicted_val > p] = 1
  
  correct_predictions = test_data$predicted_class == test_data$classification
  correct = nrow(test_data[correct_predictions,])
  total = nrow(test_data)
  accuracy = correct/total
  
  #Store the accuracy for this p cut-off
  p_acc_table_1_2 <- rbind(p_acc_table_1_2, c(p, accuracy))
}

colnames(p_acc_table_1_2) <- c("p.value", "accuracy")
write.csv(p_acc_table_1_2, "1.2_p_accuracy.csv")


#**************************************************************************************************************************************************#

#1.3 Custom classifier


#Plot histograms for span, cols_with_5, neigh5
for (f_name in c("span", "cols_with_5", "neigh5", "hollowness"))
{
  feature <- feature_data[[f_name]]
  plt <- ggplot(feature_data, aes(x=feature, fill = as.factor(classification))) +
    geom_histogram(binwidth = .2, alpha = .5, position = 'identity') +
    xlab(f_name) +
    ggtitle(sprintf("%s Histogram", f_name))
  plt
  ggsave(paste(f_name, "_histogram.png", sep = ""))
}


kfolds = 5
p <- 0.4

#The cumulative accuracy of the 5 folds, to be divided by kfolds
test_acc <- 0


shuffled_set <- feature_data[sample(nrow(feature_data)),]
shuffled_set$folds <- cut(seq(1,nrow(shuffled_set)), breaks = kfolds, labels = FALSE)

#Contingency table for use in 1.5
p_cont <- data.frame()


for (i in 1:kfolds)
{
  training_data <- shuffled_set[shuffled_set$folds != i,]
  test_data <- shuffled_set[shuffled_set$folds == i,]
  
  fit <- glm(classification ~ span + cols_with_5 + neigh5,
             data = training_data,
             family = 'binomial')
  
  #Predict accuracy over test data
  test_data$predicted_val = predict(fit, test_data, type="response")
  test_data$predicted_class = 0
  test_data$predicted_class[test_data$predicted_val > p] = 1
  
  correct_predictions = test_data$predicted_class == test_data$classification
  
  f_accuracy = nrow(test_data[correct_predictions,])/nrow(test_data)
  test_acc <- test_acc + f_accuracy
  
  #This code is for 1.5. It stores the test data labels and the predictions from the model
  
  fold_cont <- CrossTable(x = test_data$label,
                          y = test_data$predicted_class,
                          prop.chisq = F,
                          prop.r = T,
                          prop.c = F,
                          prop.t = F )
  
  #Add the results for this fold to the table and rbind it to the main table
  cont_t <- as.data.frame.matrix(fold_cont$t)
  cont_t <- cbind("label" = rownames(cont_t), cont_t)
  p_cont <- rbind(p_cont, cont_t)
  
}

#The cross validated accuracy of the model
cv_acc_1_3 <- test_acc / kfolds


#**************************************************************************************************************************************************#

#1.4

#How many correct predictions must be made by the random model?
r_size <- nrow(shuffled_set)
r_successes <- cv_acc_1_3 * nrow(shuffled_set)

#What are the odds of obtaining this number of successes?
r_binom <- pbinom(r_successes, r_size, 0.5)

#Plot the distribution
x <- 0:200
hx <- dbinom(x, r_size, 0.5)
plot(x, hx,
     xlab = "Correct Predictions",
     ylab = "Probability Density",
     main = "Binomial Distribution for a Random classification model")
abline(v = r_successes, col = "blue")
ggplot(data = NULL, aes(x = x)) +
  geom_point(aes(y = hx)) +
  geom_vline(xintercept = r_successes,
             col = "blue") +
  xlab("Correct Predictions") +
  ylab("Probability Density") +
  ggtitle("Binomial Distribution for a Random classification model")
ggsave("binom_random_class.png", scale = 1, dpi = 400)


#**************************************************************************************************************************************************#

#1.5

#Taking the table created in 1.3 in anticipation of this section, we clean the table rownames and get the total
#for each observation

rownames(p_cont) <- 1:nrow(p_cont)
p_cont <- ddply(p_cont, "label", numcolwise(sum))
p_cont$total <- rowSums(p_cont[,2:3])

write.csv(x = p_cont, file = "prediciton_test_results.csv")



library(randomForest)
library(rpart)
library(tree)
library(ipred)
library(ggplot2)
library(caret)

set.seed(3060)

#Set the datapath for getting the training data
datapath <- paste(getwd(), "doodle_data", sep = "/")
feature_col_names <- c("label", "index", "nr_pix", "height", "width", "span", "rows_wth_5", "cols_with_5", "neigh1", "neigh5", 
                       "left2tile", "right2tile", "verticalness", "top2tile", "bottom2tile", "horizontalness", "vertical3tile", "horizontal3tile", 
                       "nr_regions", "nr_eyes", "hollowness", "unused_line")

#Load Training Data from files
csv_files <- list.files(path = datapath, pattern = "_features.csv", 
                        full.names = TRUE,
                        recursive = FALSE)

#Load Training Data into dataframe
training_data <- data.frame(matrix(ncol = 22, nrow = 0))
training_data <- do.call(rbind, 
                         lapply(csv_files, function(x) {
                           csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
                         }))
colnames(training_data) <- feature_col_names


#Set the formula that will be used to build the models
features <- paste(colnames(training_data)[3:10], collapse = "+")
cl_formula <- as.formula(paste("label ~ ", features, sep = ""))


#**************************************************************************************************************************************************#

#3.1

train_sample <- training_data[sample(nrow(training_data)),]
train_size <- 0.8 * nrow(train_sample)

train_x <- cbind(train_sample[1:train_size,])

test_x <- cbind(train_sample[(train_size+1):nrow(train_sample),])


#a. Out of bag Estimation
bag_sizes <- c(25,50, 200, 400, 800)

oob_accuracies <- c()
for (bag_size in bag_sizes)
{
  t_bag <- bagging(cl_formula, data = train_x, 
                   nbagg = bag_size,
                   coob = TRUE, 
                   OOB = TRUE)
  
  pred_vals <- predict(t_bag, test_x)
  correct <- pred_vals == test_x$label
  accuracy <- nrow(test_x[correct,])/nrow(test_x)
  oob_accuracies <- c(oob_accuracies, accuracy)
}


#b. Using 5 fold cross validation
kfolds <- 5
cv_accuracies <- c()
fold_sample <- train_sample
for (bag_size in bag_sizes)
{
  fold_sample$folds <- cut(seq(1,nrow(fold_sample)),breaks=kfolds,labels=FALSE)
  cv_accuracy <- 0
  for (n in 1:kfolds)
  {
    train_bag <- fold_sample[fold_sample$fold != n,]
    test_bag <- fold_sample[fold_sample$fold == n,]   
    t_bag <- bagging(cl_formula, 
                     data = train_bag,
                     nbagg = bag_size)
    
    
    pred_vals <- predict(t_bag, test_bag)
    levels(pred_vals) <- levels(test_bag$label)
    correct <- test_bag$label == pred_vals
    accuracy <- nrow(test_bag[correct,])/nrow(test_bag)
    cv_accuracy <- cv_accuracy + accuracy
  }
  cv_accuracy <- cv_accuracy / kfolds
  cv_accuracies <- c(cv_accuracies, cv_accuracy)
}

#Visualising the results

#Table

model_results_3_1 <- data.frame(bag.size = bag_sizes, oob.accuracy = oob_accuracies, cv.accuracy = cv_accuracies)
write.csv(model_results_3_1, "3.1_results_table.csv")

#Plot

cols <- c( "Red" , "Blue")
shapes <- c("s1" = 16, "s2" = 17)
plt <- ggplot(model_results_3_1, aes(x = bag.size),
              position_dodge()) +
  geom_line(aes(y = oob.accuracy, color = "Red"), 
            lwd = 2) +
  geom_line(aes(y = cv.accuracy, color = "Blue"),
            lwd = 2) + 
  geom_point(aes(y = oob.accuracy),
             fill = "black",
             color = "Red",
             pch = 21,
             size = 4) +
  geom_point(aes(y = cv.accuracy),
             fill = "black",
             color = "Blue",
             pch = 21,
             size = 4) +
  scale_color_manual(name = "Method",
                     breaks = c("Red", "Blue"),
                     values = cols,
                     labels = c("OOB", "CV")) +
  labs(x = "Bag Size", 
       y = "Accuracy",
       color = "Method",
       title = "Model Accuracy against Bag sizes")

plt
ggsave("3.1_accuracy_plot.png")

#**************************************************************************************************************************************************#

#3.2 Classification with RandomForests
kfolds <- 5

#Variable randomForest parameters
n_trees <- seq(25,400,by = 25)
n_predictors <- c(2,4,6,8)

#Create a duplicate of the train sample so we can add the folds column without changing the original sample
f_sample <- train_sample
f_sample$folds = cut(seq(1,nrow(f_sample)),breaks=kfolds,labels=FALSE)

r_forest_results <- data.frame(n.tree = numeric(),
                               n.predictor = numeric(),
                               cv.accuracy = numeric())

for (n_tree in n_trees)
{
  for (n_predictor in n_predictors)
  {
    cv_accuracy <- 0
    #pred_features <- sample(features)
    #pred_formula <- as.formula(paste("label ~ ", sample(pred_features, n_predictor, replace = TRUE), sep = ""))
    for (i in 1:kfolds)
    {
      train_data <- f_sample[f_sample$folds != i, ]
      val_data <- f_sample[f_sample$folds == i, ]
      
      r_forest <- randomForest(cl_formula, data = train_data, 
                               ntree = n_tree,
                               mtry = n_predictor)
      pred_vals <- predict(r_forest, val_data)
      levels(pred_vals) <- levels(val_data$label)
      correct <- val_data$label == pred_vals
      accuracy <- nrow(val_data[correct,])/nrow(val_data)
      cv_accuracy <- cv_accuracy + accuracy
    }
    cv_accuracy <- cv_accuracy / kfolds
    print(sprintf("Accuracy using Random Forests with %s trees and %s predictors: %s", n_tree, n_predictor, cv_accuracy))
    r_forest_results <- rbind(r_forest_results, c(n_tree, n_predictor, cv_accuracy))
  }
}

colnames(r_forest_results) <- c("n.tree", "n.predictor", "cv.accuracy")

r_forest_plt <- ggplot(data = r_forest_results,
                       aes(x = n.tree)) +
  geom_line(aes(y = cv.accuracy, 
                color = as.factor(n.predictor)),
            lwd = 2) +
  geom_point(aes(y = cv.accuracy)) +
  labs(x = "Number of Trees", 
       y = "Cross-Validated Accuracy",
       title = "Cross-Validated Accuracy of Random Forests",
       color = "Number of Predictors")
r_forest_plt
ggsave("3.2_results_plot.png")

#**************************************************************************************************************************************************#

#3.3 

#Find the combination of tree and predictor which has the highest cross-validated accuracy
best_combn <- r_forest_results[r_forest_results$cv.accuracy == max(r_forest_results$cv.accuracy), ]
if (nrow(best_combn) > 1)
{
  best_combn <- best_combn[1,]
}
print(best_combn)
n_tree <- as.numeric(best_combn[1])
n_pred <- as.numeric(best_combn[2])

#Vector to store cross-validated accuracies of each run
cv_sample_accuracies <- c()

for (n in 1:20)
{
  f_sample <- train_sample[sample(nrow(train_sample)), ]
  f_sample$folds = cut(seq(1,nrow(f_sample)),breaks=kfolds,labels=FALSE)
  cv_accuracy <- 0
  for (i in 1:kfolds)
  {
    train_data <- f_sample[f_sample$folds != i, ]
    val_data <- f_sample[f_sample$folds == i, ]
    
    r_forest <- randomForest(cl_formula, data = train_data, 
                             ntree = n_tree,
                             mtry = n_pred)
    pred_vals <- predict(r_forest, val_data)
    levels(pred_vals) <- levels(val_data$label)
    correct <- val_data$label == pred_vals
    accuracy <- nrow(val_data[correct,])/nrow(val_data)
    cv_accuracy <- cv_accuracy + accuracy
  }
  cv_accuracy <- cv_accuracy / kfolds
  cv_sample_accuracies <- c(cv_sample_accuracies, cv_accuracy)
}

print(sprintf("Mean: %s || SD: %s", mean(cv_sample_accuracies), sd(cv_sample_accuracies)))


#**************************************************************************************************************************************************#

#3.4

#I wanted to evaluate all 7 feature subsets to find which had the best accuracy

seven_feature_combin <- combn(colnames(training_data[3:10]), 7)

seven_feature_results <- data.frame(stringsAsFactors = F)
#levels(seven_feature_results) <- colnames(training_data[3:10])

for (n in 1:ncol(seven_feature_combin))
{
  f_combo <- seven_feature_combin[,n]
  new_features <- paste(f_combo, collapse = "+")
  new_cl_formula <- as.formula(paste("label ~ ", new_features, sep = ""))
  cv_accuracy <- 0
  for (i in 1:kfolds)
  {
    train_data <- f_sample[f_sample$folds != i, ]
    val_data <- f_sample[f_sample$folds == i, ]
    
    r_forest <- randomForest(new_cl_formula, data = train_data, 
                             ntree = n_tree,
                             mtry = n_pred)
    pred_vals <- predict(r_forest, val_data)
    levels(pred_vals) <- levels(val_data$label)
    correct <- val_data$label == pred_vals
    accuracy <- nrow(val_data[correct,])/nrow(val_data)
    cv_accuracy <- cv_accuracy + accuracy
  }
  
  #Get data about this combination (missing feature, cv accuracy and difference from best accuracy of 3.3)
  removed_feature <- colnames(training_data[3:10])[! colnames(training_data[3:10]) %in%  f_combo]
  cv_accuracy <- round(cv_accuracy / kfolds, 3)
  percent_difference <- round(cv_accuracy - best_combn$cv.accuracy, 3)
  print(sprintf("CV Accuracy without using %s: %s", removed_feature, cv_accuracy))
  
  #Store results in the table
  seven_feature_results <- rbind(seven_feature_results, c(removed_feature, cv_accuracy, percent_difference), stringsAsFactors = F)
}

colnames(seven_feature_results) <- c("removed.feature", "cv.accuracy", "percent.difference")
write.csv(seven_feature_results, "3.4_model_results.csv")


#I decided to remove the cols_with_5 feature based on the previous loop

new_features <- paste(seven_feature_combin[,3], collapse = "+")
new_cl_formula <- as.formula(paste("label ~ ", new_features, sep = ""))
accuracies_3_4 <- c()
for (n in 1:20)
{
  f_sample <- train_sample[sample(nrow(train_sample)), ]
  f_sample$folds = cut(seq(1,nrow(f_sample)),breaks=kfolds,labels=FALSE)
  cv_accuracy <- 0
  for (i in 1:kfolds)
  {
    train_data <- f_sample[f_sample$folds != i, ]
    val_data <- f_sample[f_sample$folds == i, ]
    
    r_forest <- randomForest(new_cl_formula, data = train_data, 
                             ntree = n_tree,
                             mtry = n_pred)
    pred_vals <- predict(r_forest, val_data)
    levels(pred_vals) <- levels(val_data$label)
    correct <- val_data$label == pred_vals
    accuracy <- nrow(val_data[correct,])/nrow(val_data)
    cv_accuracy <- cv_accuracy + accuracy
  }
  accuracies_3_4 <- c(accuracies_3_4, cv_accuracy / kfolds)
}
accuracies_3_4
print(sprintf("Mean: %s || SD: %s", mean(accuracies_3_4), sd(accuracies_3_4)))
print(sprintf("Difference %s", mean(accuracies_3_4) - best_combn$cv.accuracy))


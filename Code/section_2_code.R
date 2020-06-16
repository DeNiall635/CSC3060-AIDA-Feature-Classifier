library(ggplot2)
library(gmodels)
library(caret)
library(e1071)
library(class)
library(plyr)

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
training_data <- do.call(rbind, lapply(csv_files, function(x) {
  csv_features <- read.csv(x, sep = "\t", header = FALSE, col.names = feature_col_names)
}))
colnames(training_data) <- feature_col_names

#**************************************************************************************************************************************************#

#2.1 KNN model for features 1-8

#Create training and test samples
training_sample <- training_data[sample(nrow(training_data)),]
train_size <- 0.8 * nrow(training_sample)

train_x <- cbind(training_sample[1:train_size, 3:10])

test_x <- cbind(training_sample[(train_size+1):nrow(training_sample), 3:10])


cl <- training_sample$label[1:train_size]

#Vectors to store the values for k and it's associated accuracy for each iteration
odd_k <- c()
knn_acc <- c()


for (n in 1:59)
{
  if (n %% 2 == 1)
  {
    knn_pred <- knn(train_x, 
                    test_x,
                    cl,
                    k = n)
    odd_k <- c(odd_k, n)
    accuracy <- mean(knn_pred == training_sample$label[(train_size+1):nrow(training_sample)])
    knn_acc <- c(knn_acc, accuracy)
  }
}

#Store the results in a table
knn_table <- data.frame("k"= odd_k, "accuracy" = knn_acc)


#**************************************************************************************************************************************************#

#2.2 KNN using 5 fold Cross validation for same 8 features
kfolds = 5

#Set the predictor features
cl_fs <- colnames(training_sample)[3:10]

#Vectors to store the inverse k value (1/k) and the associated cross-validated accuracy
cv_err <- c()
inv_k <-c()

for (n in 1:59)
{
  if (n %% 2 == 1)
  {
    training_sample$folds <- cut(seq(1,nrow(training_sample)), breaks = kfolds, labels = FALSE)
    cv_accuracy = 0
    for (i in 1:kfolds)
    {
      train_items = training_sample[training_sample$folds != i,]
      test_items = training_sample[training_sample$folds == i,]
      
      knn_pred = knn(train_items[,cl_fs],
                     test_items[,cl_fs],
                     train_items$label,
                     k = n)
      
      correct_items = nrow(test_items[knn_pred == test_items$label,])
      test_accuracy = correct_items/nrow(test_items)
      
      cv_accuracy = cv_accuracy + test_accuracy
    }
    cv_accuracy = cv_accuracy/kfolds
    cv_err <- c(cv_err, cv_accuracy)
    inv_k <- c(inv_k, 1/n)
  }
}

#Append the cross-validated accuracies and inverse k values to the dataframe created in 2.1
knn_table <- cbind(knn_table, cv_err, inv_k)
colnames(knn_table)[3] <- "cv.accuracy"
colnames(knn_table)[4] <- "1.div.k"

#Plot the accuracies for each method against the inverse of k

#X-axis graph ticks
k_ticks <- c(0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0)

acc_plt <- ggplot(knn_table, aes(x = knn_table$`1.div.k`),
                  position_dodge()) +
  geom_line(aes(y = knn_table$cv.accuracy),
            colour = "red",
            lwd =1.25,
            lty = "longdash") +
  geom_line(aes(y = knn_table$accuracy),
            colour = "blue",
            lwd = 1.25,
            lty = "longdash") +  
  xlab("1/K") +
  ylab("Accuracy Rate") + 
  geom_point(aes(y = knn_table$cv.accuracy, fill = "blue"),
             col = "black",
             pch = 21,
             size = 2.5) +
  geom_point(aes(y = knn_table$accuracy, fill = "red"),
             col = "black",
             pch = 21,
             size = 2.5) +
  scale_x_log10(breaks = k_ticks) +
  scale_fill_discrete(name = "",
                      breaks = c("blue", "red"),
                      labels = c("Cross-Val", "Original"))
acc_plt
ggsave("2.2_knn_acc.png", scale = 1, dpi = 400)


#**************************************************************************************************************************************************#

#2.3 Evaluating the prediciton accuracy of the model for each object

#Using a k value of 5 based on the results table of the cross validation models
#I will use Cross tables to represent the results 

#Empty dataframe which will collect the results from each fold
c_table_2_3 <- data.frame()

for (i in 1:kfolds)
{
  train_items = training_sample[training_sample$folds != i,]
  test_items = training_sample[training_sample$folds == i,]
  
  knn_pred = knn(train_items[,cl_fs],
                 test_items[,cl_fs],
                 train_items$label,
                 k = 5)
  ct <- CrossTable(x = test_items$label,
                   y = knn_pred,
                   prop.chisq = FALSE,
                   prop.c = FALSE)
  
  #Store the results from each each fold into the empty table
  cont_t <- as.data.frame.matrix(ct$t)
  cont_t <- cbind("label" = rownames(cont_t), cont_t)
  c_table_2_3 <- rbind(c_table_2_3, cont_t)
}

#Minor fixes to the table to improve its readability
rownames(c_table_2_3) <- 1:nrow(c_table_2_3)
c_table_2_3 <- ddply(c_table_2_3, "label", numcolwise(sum))
c_table_2_3$total <- rowSums(c_table_2_3[,2:9])

write.csv(x = c_table_2_3, file = "2.3_knn_cross_table.csv")


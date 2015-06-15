---
title: "Practical Machine Learning Project Report"
output: html_document
fontsize: 11pt
---

### **Introduction**
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did the exercise.

#### **Data Preprocessing**
```{r}
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

#Download the Data
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Read the Data
trainRaw <- read.csv(url(trainUrl))
testRaw <- read.csv(url(testUrl))
dim(trainRaw)
dim(testRaw)

```

The training data set contains 19622 observations and 160 variables, while the testing data set contains 20 observations and 160 variables. The “classe” variable in the training set is the outcome to predict.

#### **Clean the data**
In this step, we will clean the data and get rid of observations with missing values as well as some meaningless variables.

```{r}
sum(complete.cases(trainRaw))
```
First, we remove columns that contain NA missing values.
```{r}
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```

Next, we get rid of some columns that do not contribute much as predictors.
```{r}
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```

Now, the cleaned training data set contains 19622 observations and 53 variables, while the testing data set contains 20 observations and 53 variables. The “classe” variable is still in the cleaned training set.

#### **Slice the data**
Then, we can split the cleaned training set into a pure training data set (70%) and a validation data set (30%). We will use the validation data set to conduct cross validation in future steps.**Remove zero covariates** if at all present in predictors(which means less meaning predictors are removed).Here no zerovariance hence all predictors are included.
```{r}
set.seed(22252) # For reproducibile purpose
inTrain <- createDataPartition(trainCleaned$classe, p=0.70, list=F)
trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
#Removing zero covariates
nearZeroVar(trainData,saveMetrics=F)
```
#### **Data Modeling**
We fit a predictive model for activity recognition using **Random Forest algorithm** because it automatically selects important variables and is robust to correlated covariates & outliers in general. We will use **10-fold cross validation**(default) when applying the algorithm.
```{r}
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=trainControl(method="cv"))
modelRf
```

**Then, we estimate the performance of the model on the validation data set and out-of-sample error.**
```{r}
predictRf <- predict(modelRf, testData)
confusionMatrix(testData$classe, predictRf)

accuracy <- postResample(predictRf, testData$classe)
accuracy

oose <- 1 - as.numeric(confusionMatrix(testData$classe, predictRf)$overall[1])
oose
```

So, the estimated accuracy of the model is `r accuracy[1]` % and the estimated out-of-sample error is `r oose` %

#####**Predicting for Test Data Set**
Now, we apply the model to the original testing data set downloaded from the data source. 

```{r}
result <- predict(modelRf, testCleaned)
result
```
### **Appendix: Figures**
plot (below) shows the importance of each variable computed in the random forest model.

```{r}
plot(varImp(modelRf))
```

#### **Tree Visualization**
```{r}
treeModel <- rpart(classe ~ ., data=trainData , method="class")
prp(treeModel) # fast plot
```

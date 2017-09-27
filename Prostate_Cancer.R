library(class)
library(caret)
library(e1071)

prostate_Cancer <- read.csv("Prostate_Cancer.csv")

## Convert the dependent var to factor. Normalize the numeric variables 
num.vars <- sapply(prostate_Cancer, is.numeric)
prostate_Cancer[num.vars] <- lapply(prostate_Cancer[num.vars], scale)


## Selecting only 3 numeric variables for this demostration, just to keep things simple
myvars <- c("radius", "texture", "perimeter", "area", "smoothness", "compactness", "symmetry", "fractal_dimension")
prostate_Cancer.subset  <- prostate_Cancer[myvars]
summary(prostate_Cancer[myvars])

## Let's predict on a test set of 100 observations. Rest to be used as train set.
set.seed(100) 
test <- 1:20
train.prostate_Cancer <- prostate_Cancer.subset[-test,]
test.prostate_Cancer <- prostate_Cancer.subset[test,]
train.def <- prostate_Cancer$diagnosis_result[-test]
test.def <- prostate_Cancer$diagnosis_result[test]

## Let's use k values (no of NNs) as 1, 5 and 20 to see how they perform in terms of correct proportion of classification and success rate. The optimum k value can be chosen based on the outcomes as below...
knn.1 <-  knn(train.prostate_Cancer, test.prostate_Cancer, train.def, k=1)
knn.5 <-  knn(train.prostate_Cancer, test.prostate_Cancer, train.def, k=5)
knn.20 <- knn(train.prostate_Cancer, test.prostate_Cancer, train.def, k=20)

## Let's calculate the proportion of correct classification for k = 1, 5 & 20 
100 * sum(test.def == knn.1)/100  # For knn = 1
100 * sum(test.def == knn.5)/100  # For knn = 5
100 * sum(test.def == knn.20)/100 # For knn = 20


## We should also look at the success rate against the value of increasing K.
table(knn.1 ,test.def)
table(knn.5 ,test.def)
table(knn.20 ,test.def)

## Evaluation of the predictive performance
confusionMatrix(knn.1, test.def)
confusionMatrix(knn.5, test.def)
confusionMatrix(knn.20, test.def)

library(ElemStatLearn)
set = train.prostate_Cancer
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Radius', 'Texture')
y_grid = knn(train = train.prostate_Cancer,
             test = grid_set,
             cl = train.prostate_Cancer,
             k = 5)
plot(set,
     main = 'K-NN (Training set)',
     xlab = 'Radius', ylab = 'Texture',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set == 1, 'green4', 'red3'))
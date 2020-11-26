##upload and attach dataset
Happiness <- read.csv("Happiness_2016.csv")
attach(Happiness)
##explore dataset
dim(Happiness)
names(Happiness)
sapply(Happiness, class)
head(Happiness)

##removing variables so that only Happiness score, Economy, Family, 
##Health, Freedom, Trust and Generosity remain
Happiness2 <- Happiness[,-c(1,2,3,5,6,13)] 
head(Happiness2)

HighHappiness = ifelse(Happiness.Score<=6,"No","Yes")
Happiness3=data.frame(Happiness2,HighHappiness)
head(Happiness3)

Happinessnew=Happiness3[,-1] ##removing happiness score variable
names(Happinessnew)

install.packages("tidyverse")
install.packages("ggplot2")
library(tidyverse)
library(ggplot2)

ggplot(Happinessnew, aes(x=HighHappiness, y=Economy, fill=HighHappiness)) +
  geom_boxplot()+
 scale_fill_brewer(palette="Paired") + theme_minimal()
ggplot(Happinessnew, aes(x=HighHappiness, y=Family, fill=HighHappiness)) +
  geom_boxplot()+
  scale_fill_brewer(palette="Paired") + theme_minimal()
ggplot(Happinessnew, aes(x=HighHappiness, y=Health, fill=HighHappiness)) +
  geom_boxplot()+
  scale_fill_brewer(palette="Paired") + theme_minimal()
ggplot(Happinessnew, aes(x=HighHappiness, y=Freedom, fill=HighHappiness)) +
  geom_boxplot()+
  scale_fill_brewer(palette="Paired") + theme_minimal()
ggplot(Happinessnew, aes(x=HighHappiness, y=Trust, fill=HighHappiness)) +
  geom_boxplot()+
  scale_fill_brewer(palette="Paired") + theme_minimal()
ggplot(Happinessnew, aes(x=HighHappiness, y=Generosity, fill=HighHappiness)) +
  geom_boxplot()+
  scale_fill_brewer(palette="Paired") + theme_minimal()





##variance co variance matrix
pairs(Happiness2, panel=panel.smooth)
##correlation
cor(Happiness2)

set.seed(6)
tr1 = sample(1:157, 82)
train = Happiness2[tr1, ]
dim(train)

##Multiple linear regression
model= lm(Happiness.Score~Economy+Family+Health+Freedom
          +Trust+Generosity, data=Happiness2, subset=tr1)
summary(model)
##calculating MSE of observations in validation set
mean((Happiness.Score -predict (model,Happiness2))[-tr1]^2)

##remove generosity and trust variables
model2 = lm(Happiness.Score~Economy+Family+Health+Freedom, 
            data=Happiness2, subset=tr1)
summary(model2)
mean((Happiness.Score -predict (model2,Happiness2))[-tr1]^2)

##polynomical
poly = lm(Happiness.Score~Freedom+I(Freedom^2)+I(Freedom^3), 
          data=Happiness2, subset=tr1)
summary(poly)

##confidence intervals
confint(model2)

##overall accuracy of model
anova(model2)
qf(0.95,2,197)

##sampling full data source to create testing data set
testing_set = Happiness2[sample(nrow(Happiness2),size=117,replace=TRUE),] 
##sample of approx. 75% of original data

##predicted values and residuals
predict(model2,testing_set)
resid(model2,testing_set)
predict(model2,testing_set,interval='confidence')

##plot residuals against the predicted values
plot(predict(model2,data="testing_set"),resid(model2,data="testing_set"), 
     xlab = "Fitted Values", ylab = "Residuals")
hist(resid(model2,data=testing_set), main = paste("Histogram of Residuals"), 
     xlab = "Residuals")

##model checking
par(mfrow=c(2,2))
plot(model2)

##Decision Tree (Classification)
library(tree)
##Transfer happiness score variable from continuous to categorical
HighHappiness = ifelse(Happiness.Score<=6,"No","Yes")
Happiness3=data.frame(Happiness2,HighHappiness)
Happinessnew=Happiness3[,-1] ##removing happiness score variable

##Fit classification tree for the full data set
tree_happy <- tree(HighHappiness~.,Happinessnew)
plot(tree_happy)
text(tree_happy, pretty=0)

##Test model accuracy
summary(tree_happy)

##Fit classification tree for training data set
set.seed(10)
tr.tree <- sample(1:157, 79)
test.tree <- Happinessnew[-tr.tree,]
train.tree <- tree(HighHappiness~.,Happinessnew, subset=tr.tree)
plot(train.tree)
text(train.tree, pretty=0)

##Test model accuracy
##Predict High Happiness for test data
tree_predict =predict(train.tree,test.tree,type="class")

##Misclassification matrix
test_high <- HighHappiness[-tr.tree] ##high happiness in test data set
table(tree_predict,test_high)

##misclassification rate
matrix <- table(tree_predict,test_high)
misclassification_rate <- ((matrix[1,2]+matrix[2,1])/sum(matrix))
cat("Misclassification error rate is ",misclassification_rate)

##cross validation plot
set.seed(10)
cv_Happiness <- cv.tree(train.tree, FUN=prune.misclass)
names(cv_Happiness)
plot(cv_Happiness$size, cv_Happiness$dev, type="b")
cv_Happiness

##obtain best tree by pruning
prune_happy <- prune.misclass(train.tree, best=3)
prune_happy
plot(prune_happy)
text(prune_happy, pretty=0)

##Predict fpr pruned model
pruned_predict =predict(prune_happy,test.tree,type="class")

##Misclassification matrix
test_high_pruned <- HighHappiness[-tr.tree] ##high happiness in test data set
table(pruned_predict,test_high)

##misclassification rate
matrix_pruned <- table(pruned_predict,test_high)
misclassification_rate_pruned <- ((matrix_pruned[1,2]+matrix_pruned[2,1])/
                                    sum(matrix_pruned))
cat("Misclassification error rate is ",misclassification_rate_pruned)

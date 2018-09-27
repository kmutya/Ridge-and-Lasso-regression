
#Libraries to be used
library(glmnet) #regularized regression package
library(ggplot2) #for plotting
library(caret) #hot one encoding
library(e1071) #skewess function
library(gridExtra) #grid mapping

#Importing King county house sales dataset locally
getwd()
setwd("/Users/apple/Google Drive/github/Regularization")
data = read.csv('kc_house_data.csv')
data = subset(data, select = -c(id, date))
data$floors = as.character(data$floors)
data$zipcode = as.character(data$zipcode)
str(data)
dim(data) #21613 x 19
sum(is.na(data)) #No missing data

#Plotting histograms to see distribution of all numeric variables

#gather all non charecter features
feature_class = sapply(colnames(data),function(x){class(data[[x]])})
numeric_features = names(feature_class[feature_class != "character"])
numeric_features

#Plotting using ggplot and fixing grid using gridExtra
dist_list = list()
for (i in numeric_features){
  dist_list[[i]] = ggplotGrob(ggplot(data = data, aes_string(x = i)) + geom_histogram(aes(x = data[i])) + theme_grey())
}

grid.arrange(dist_list[[1]],dist_list[[2]],dist_list[[3]],dist_list[[4]], dist_list[[5]],
             dist_list[[6]],dist_list[[7]],dist_list[[8]],dist_list[[9]],dist_list[[10]], 
             dist_list[[11]], dist_list[[12]], ncol=3)

#Removing skew for all numeric features using a power transformation (log) and hot one encoding categorical variables

# determine skew for each numeric feature
skewed_feats = sapply(numeric_features, function(x) {skewness(data[[x]])})
skewed_feats
#remove skew greater than 1
rem_skew = skewed_feats[skewed_feats>1]
rem_skew
for (i in names(rem_skew)){
  data[[i]] = log(data[[i]]+1) # +1 as we have many 0 values in many columns
}
head(data)

#hot one encoding
categorical_feats = names(feature_class[feature_class == "character"])
categorical_feats
dummies = dummyVars(~., data[categorical_feats]) #from library caret
categorical_1_hot = predict(dummies, data[categorical_feats])

#Create master file and perform training-testing split

#Combining files
master_data = cbind(data[numeric_features], categorical_1_hot)

#Creating a training and testing split
l = round(0.7*nrow(master_data))
set.seed(7)
seq_rows = sample(seq_len(nrow(master_data)), size = l)
data_train = master_data[seq_rows,]
data_test = master_data[-seq_rows,]

#create matrices
data_train_x = as.matrix(data_train[,2:93])
data_train_y = as.matrix(data_train$price)
data_test_x = as.matrix(data_test[,2:93])
data_test_y = as.matrix(data_test$price)

#1. k-fold cv (k = 10 by default)
ridge_cv = cv.glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 0 #0 for ridge
)
plot(ridge_cv)

min(ridge_cv$cvm) #lowest MSE
ridge_cv$lambda.min #lambda for lowest MSE
min = ridge_cv$lambda.1se #selecting the 1st se from lowest

#2. Final model
#visualization
ridge = glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 0
)
plot(ridge, xvar = "lambda")
abline(v = log(ridge_cv$lambda.1se), col = "red", lty = "dashed") #lambda value we picked

#building model using the seleceted lambda value
ridge_min = glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 0, lambda = min
)
#Visualizing important variables

#function to return dataframe of coefficents without an intercept
coefficents = function(coefi, n){
  #coefi -> is the beta output of glmnet function
  #n -> is the desired number of features to be plotted
  #returns features coerced with values ordered in desc by abs value
  coef_v = as.matrix(coefi)
  as.data.frame(coef_v)
  colnames(coef_v)[1] = 'values'
  coef_f =  as.data.frame(dimnames(coef_v)[[1]])
  coef_final = cbind(coef_f,coef_v)
  colnames(coef_final)[1] = 'features'
  coef_final = coef_final[order(-abs(coef_final$values)),] 
  coef_final$values = round(coef_final$values, 2)
  coef_final = coef_final[1:n,]
  return(coef_final)
}

ggplot(coefficents(ridge_min$beta, 25), aes(x=features, y=values, label=values)) + 
  geom_point(stat='identity', fill="Black", size=6)  +
  geom_segment(aes(y = 0, 
                   x = features, 
                   yend = values, 
                   xend = features), 
               color = "Black") +
  geom_text(color="white", size=2) +
  labs(title="Top 25 influential variables", y = 'Coefficient Value') + 
  ylim(-1, 1) +
  coord_flip()

#predicting on test set
y_pred = predict(ridge_min, data_test_x)
#function to compute total sum of squares
r_sq = function(y, pred_y){
  #y -> Actual value of y in the test set
  #pred_y -> predicted y value
  tss = sum((y - mean(y))^2)
  rss = sum((pred_y - y)^2)
  return(reslut = 1 - (rss/tss))
}

r_sq(data_test_y, y_pred) #0.8841862

##################LASSO###############
#1. k-fold cv (k = 10 by default)
lasso_cv = cv.glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 1 #1 for lasso
)
plot(lasso_cv)

min(lasso_cv$cvm) #lowest MSE
lasso_cv$lambda.min #lambda for lowest MSE
min_l = lasso_cv$lambda.1se #selecting the 1st se from lowest

#2. Final model
#visualization
lasso = glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 1
)
plot(lasso, xvar = "lambda")
abline(v = log(lasso_cv$lambda.1se), col = "red", lty = "dashed") #lambda value we picked

#building model using the seleceted lambda value
lasso_min = glmnet(
  x = data_train_x,
  y = data_train_y,
  alpha = 1, lambda = min_l
)

#number of non zero coeff
length(lasso_min$beta[lasso_min$beta!=0]) #82

#Visualizing important variables
#let's utilize the function we created earlier as an input to the plot
# Diverging chart
ggplot(coefficents(lasso_min$beta, 82), aes(x=features, y=values, label=values)) + 
  geom_point(stat='identity', fill="Black", size=6)  +
  geom_segment(aes(y = 0, 
                   x = features, 
                   yend = round(values,2), 
                   xend = features), 
               color = "Black") +
  geom_text(color="white", size=2) +
  labs(title="Top influential variables", y = 'Coefficient Value') + 
  ylim(-1, 1) +
  coord_flip()

#predicting lasso on test set
y_pred_l = predict(lasso_min, data_test_x)
#computing r_squared
r_sq(data_test_y, y_pred_l) #0.8858732




#Import clean dataset with missing data

df = read.csv("clean_df2.csv")
df

install.packages("BaylorEdPsych")
install.packages("mvnmle")


library("BaylorEdPsych")

LittleMCAR(df)

#Check multivariate outliers on clean data still with outliers

df1 = read.csv("task1_clean_replaced_NaNs.csv")

df1

install.packages("psych")

library(psych)

#use mahalanobis function

md <- mahalanobis(df1, center = colMeans(df1, na.rm = T), cov = cov(df1, use = "complete.obs"))

alpha <- .001
cutoff <- (qchisq(p = 1 - alpha, df = ncol(df1)))
outlierID.mah=df1$X[md > cutoff]

# remove NAs
outlierID.mah=outlierID.mah[!is.na(outlierID.mah)]
outlierID.mah

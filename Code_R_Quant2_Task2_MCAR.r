# Task 2 MCAR

#Run Little's MCAR test to see if data is missing completely at random

df = read.csv("Task2_data_NaN.csv")
df

install.packages("BaylorEdPsych")
install.packages("mvnmle")

library("BaylorEdPsych")

LittleMCAR(df)



df = read.csv("Task2_data_NaN.csv")
df

install.packages("BaylorEdPsych")
install.packages("mvnmle")

library("BaylorEdPsych")

LittleMCAR(df)



#Run Little MCAR test; import data still with missing values

df = read.csv("Task3_data_NaN.csv")
df

#Install required packages

install.packages("BaylorEdPsych")
install.packages("mvnmle")

library("BaylorEdPsych")

LittleMCAR(df)

#OK, do rest of data cleaning in Python

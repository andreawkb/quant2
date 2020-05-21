#Import cleaned dataset with no outliers

df = read.csv("task1_data_clean_no_outliers.ready.csv")
print(df)

reg = lm(GovSupport~NegEmot, data=df)

plot(reg)

summary(reg)

#Multicollinearity test, create dataframe with the three independent variables to run diagnostics

df2 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism)

#Run correlation matrix first of all

cor(df2)

#Correlation looks OK, everything <0.8. Now need to run multiple regression model first before I can run the other multicollinearity diagnostics. 

reg2 = lm(GovSupport~NegEmot+Egalitarianism+Individualism, data=df)
summary(reg2)

#Check assumptions for 2nd regression model

plot(reg2)

#Check assumptions for 3rd regression model

reg3 = lm(GovSupport~NegEmot+Egalitarianism+Individualism+AGE+GENDER, data=df)
summary(reg3)

#Check multicollinearity for reg3 independent variables

df3 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism, df$GENDER, df$AGE)

cor(df3)

plot(reg3)

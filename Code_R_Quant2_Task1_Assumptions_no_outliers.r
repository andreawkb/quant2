df = read.csv("task1_data_clean_no_outliers.ready.csv")
print(df)

reg = lm(GovSupport~NegEmot, data=df)

plot(reg)

summary(reg)

df2 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism)

cor(df2)

reg2 = lm(GovSupport~NegEmot+Egalitarianism+Individualism, data=df)
summary(reg2)

plot(reg2)

reg3 = lm(GovSupport~NegEmot+Egalitarianism+Individualism+AGE+GENDER, data=df)
summary(reg3)

df3 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism, df$GENDER, df$AGE)

cor(df3)

plot(reg3)

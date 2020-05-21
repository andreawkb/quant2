df = read.csv("task1_data_clean_ready.csv")
print(df)

reg = lm(GovSupport~NegEmot, data=df)

plot(reg)

plot(reg, which= c(3))


plot(reg, which=2)

summary(reg)

df2 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism)

cor(df2)

reg2 = lm(GovSupport~NegEmot+Egalitarianism+Individualism, data=df)
summary(reg2)

plot(reg2, which= c(3))

plot(reg2, which=2)

reg3 = lm(GovSupport~NegEmot+Egalitarianism+Individualism+AGE+GENDER, data=df)
summary(reg3)

df3 = data.frame(df$NegEmot, df$Egalitarianism, df$Individualism, df$GENDER, df$AGE)

cor(df3)

plot(reg3, which=2)

plot(reg3, which= c(3))

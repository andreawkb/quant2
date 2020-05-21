df = read.csv("Task2_data_clean_withOutliers_ready.csv")

df

reg = lm(Energy~AutSup, data=df)

summary(reg)

plot(reg, which= c(3))

plot(reg, which=2)

df2 = data.frame(df$AutSup, df$IntMot)

cor(df2)

reg2 = lm(Energy~AutSup+IntMot, data=df)

plot(reg2, which=2)

plot(reg2, which= c(3))

reg3 = lm(IntMot~AutSup, data=df)

plot(reg3, which= c(3))
plot(reg3, which=2)

reg4 = lm(Energy~IntMot, data=df)

plot(reg4, which= c(3))
plot(reg4, which=2)

df5 = read.csv('Task2_data_clean_no_outliers_ready.csv')

reg5 = lm(Energy~AutSup, data=df5)
summary(reg5)

plot(reg5, which= c(3))
plot(reg5, which=2)

reg6 = lm(Energy~AutSup+IntMot, data=df5)
plot(reg6)

reg7 = lm(IntMot~AutSup, data=df5)
plot(reg7)

reg8 = lm(Energy~IntMot, data=df5)
plot(reg8)

df10 = data.frame(df5$AutSup, df5$IntMot)
cor(df10)

install.packages("lavaan", dependencies=TRUE)

set.seed(20200510)
library(foreign)
library(lavaan)

model <- ' # direct effect
             Energy ~ c*AutSup
           # mediator
             IntMot ~ a*AutSup
             Energy ~ b*IntMot
           # indirect effect (a*b)
             ab := a*b
           # total effect
             total := c + (a*b)
         '
fit <- sem(model, data = df)
summary(fit)

model <- ' # direct effect
             Energy ~ c*AutSup
           # mediator
             IntMot ~ a*AutSup
             Energy ~ b*IntMot
           # indirect effect (a*b)
             ab := a*b
           # total effect
             total := c + (a*b)
         '
fit <- sem(model, data = df5)
summary(fit)



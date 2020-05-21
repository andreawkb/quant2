## Task 2 - Assumptions testing & Primary Analysis (Mediation Model)

df = read.csv("Task2_data_clean_withOutliers_ready.csv")

df

#Total effect (c path) regression y = Energy, x = AutSup

reg = lm(Energy~AutSup, data=df)

summary(reg)

#Test assumptions for this model

plot(reg, which= c(3))

plot(reg, which=2)

#Multicollinearity test for multiple regression (<8) 

df2 = data.frame(df$AutSup, df$IntMot)

cor(df2)

#Multiple Regression to control for intrinsic motivation (c' path): x=AutSupport, y=Energy, m=IntMot.

#Test assumptions.

reg2 = lm(Energy~AutSup+IntMot, data=df)

plot(reg2, which=2)

plot(reg2, which= c(3))

#Regression - a path (x=AutSup, y=IntMot)

reg3 = lm(IntMot~AutSup, data=df)

#test assumptions, a path regression

plot(reg3, which= c(3))
plot(reg3, which=2)

#Regression - b path (x=IntMot, y=Energy)

#Test assumptions.

reg4 = lm(Energy~IntMot, data=df)

plot(reg4, which= c(3))
plot(reg4, which=2)

## Assumptions - no outliers

#Import cleaned dataset with outliers removed (in Python)

df5 = read.csv('Task2_data_clean_no_outliers_ready.csv')

#c path

reg5 = lm(Energy~AutSup, data=df5)
summary(reg5)

plot(reg5, which= c(3))
plot(reg5, which=2)

#c' path

reg6 = lm(Energy~AutSup+IntMot, data=df5)
plot(reg6)

#a path

reg7 = lm(IntMot~AutSup, data=df5)
plot(reg7)

#b path

reg8 = lm(Energy~IntMot, data=df5)
plot(reg8)

#Check if independent variables are correlated (multicollinearity)

df10 = data.frame(df5$AutSup, df5$IntMot)
cor(df10)

## Mediation Model (dataset with outliers)

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

## Mediation Model (dataset without outliers)

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



install.packages("psych")

library(psych)

df = read.csv("Task3_data_clean.csv")

df

describe(df)

hist(df$PER1)
hist(df$PER2)
hist(df$PER3)

boxplot(df$PER1)
boxplot(df$PER2)
boxplot(df$PER3)

boxplot(df$RUM1)
boxplot(df$RUM2)
boxplot(df$RUM3)

boxplot(df$EX1)
boxplot(df$EX2)

boxplot(df$EX3)

boxplot(df$EX4)

md = mahalanobis(df, center = colMeans(df, na.rm = T), cov = cov(df, use = "complete.obs"))

alpha <- .001
cutoff <- (qchisq(p = 1 - alpha, df = ncol(df)))
outlierID.mah=df$X[md > cutoff]

# remove NAs
outlierID.mah=outlierID.mah[!is.na(outlierID.mah)]
outlierID.mah

winsorize <- function(x, probs = NULL, cutpoints = NULL , replace = c(cutpoints[1], cutpoints[2]), verbose = TRUE){
  dummy = is.integer(x)
  if (!is.null(probs)){
    stopifnot(is.null(cutpoints))
    stopifnot(length(probs)==2)
    cutpoints <- quantile(x, probs, type = 1, na.rm = TRUE)
  } else if (is.null(cutpoints)){
    l <- quantile(x, c(0.25, 0.50, 0.75), type = 1, na.rm = TRUE) 
    cutpoints <- c(l[2]-3*(l[3]-l[1]), l[2]+3*(l[3]-l[1]))  ### Default was 5*IQR but has been changed to 3*IQR
  } else{
    stopifnot(length(cutpoints)==2)
  }
  if (is.integer(x)) cutpoints <- round(cutpoints)
  bottom <- x < cutpoints[1]
  top <- x > cutpoints[2]
  if (verbose){
    length <- length(x)
    message(paste(100*sum(bottom, na.rm = TRUE)/length,"% observations replaced at the bottom"))
    message(paste(100*sum(top, na.rm = TRUE)/length,"% observations replaced at the top"))
  }
  x[bottom] <- replace[1]
  x[top] <- replace[2]
  if (dummy){
    x <- as.integer(x)
  }
  x
}

df1 = apply(df,2,winsorize)

describe(df1)

install.packages("lavaan", dependencies = TRUE)
library(lavaan)
example(cfa)

model = 'Perfectionism =~ PER1 + PER2 + PER3
 Rumination =~ RUM1 + RUM2 + RUM3
 Exhaustion =~ EX1 + EX2 + EX3 + EX4'

fit = cfa(model, data=df1)

summary(fit, fit.measures=TRUE)

install.packages("dplyr")

install.packages("tidyr")

library(dplyr)
library(tidyr)

install.packages("knitr")

library(knitr)

parameterEstimates(fit, standardized=TRUE) %>% 
  filter(op == "=~") %>% 
  select('Latent Factor'=lhs, Indicator=rhs, B=est, SE=se, Z=z, 'p-value'=pvalue, Beta=std.all) %>% 
  kable(digits = 3, format="pandoc", caption="Factor Loadings")

model_sem = 'Perfectionism =~ PER1 + PER2 + PER3
 Rumination =~ RUM1 + RUM2 + RUM3
 Exhaustion =~ EX1 + EX2 + EX3 + EX4
#regressions
 Rumination ~ Perfectionism
 Exhaustion ~ Rumination'

fit_sem = sem(model_sem, data=df1)

summary(fit_sem, fit.measures=TRUE)

inspect(fit_sem, 'r2')

#write.csv(df1, "C:\\iCloudDrive\\Documents\\Task3_data_winsorized.csv", row.names = FALSE)



library(MASS)
library(dplyr)
library(readxl)
library(gtsummary)
library(ggplot2)
library(tidyr)
library(sjPlot)

setwd('/Users/gabrielevezzani/Desktop/Dottorato/DH_Accollo')
data <- read_excel("data/dataset_processed.xlsx", sheet = "Sheet1")
data$year <- format(as.Date(data$date, format = "%B %d, %Y"), "%Y")
data <- data %>% 
  distinct(link, .keep_all = TRUE) %>% 
  mutate(rating = as.factor(rating)) %>%
  mutate(year = as.numeric(year))


model <- polr(rating~year, data=data)
summ <- summary(model)

ctable <- coef(summ)
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable <- cbind(ctable, "p value" = p)




file_path <- 'results/regression_ratings.txt'
sink(file_path, append = FALSE)
print(summ)
print(ctable)
cat('\n\nP VALUE OF CHI SQUARED TEST:\n')
print(p)
sink()

plot = plot_model(model, type = "eff", terms = "year")

ggsave('results/probabilities.png', plot)



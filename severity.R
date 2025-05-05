# install.packages("readxl", repos='https://cloud.r-project.org')
# install.packages("openxlsx", repos='https://cloud.r-project.org')

library(readxl)
library(openxlsx)

#my_data <- read_excel("data/joined_data.xlsx")
my_data <- read_excel("data/evaluation_data.xlsx")

model <- readRDS("glm_severity_model", refhook = NULL)

new_data <- my_data[, c("credit_band", "ownership", "loyalty", "claims_history")]

summary(model)

dummy.coef(model)

prediction <- predict(model, data=my_data, type = "response")

prediction

df <- data.frame(Prediction = matrix(prediction, ncol = 1, byrow = TRUE))

write.xlsx(df, "data/evaluation_severity.xlsx")
library(readxl)
library(openxlsx)

my_data <- read_excel("C:/Users/Jacks/Desktop/Co-operators-Case-Competition/data/joined_data.xlsx")

model <- readRDS("C:/Users/Jacks/Desktop/Co-operators-Case-Competition/glm_severity_model", refhook = NULL)

new_data <- my_data[, c("credit_band", "ownership", "loyalty", "claims_history")]

summary(model)

dummy.coef(model)

prediction <- predict(model, data=my_data, type = "response")

prediction

df = data.frame(matrix(prediction, ncol = 1, byrow = T ))

write.xlsx(df, "C:/Users/Jacks/Desktop/Co-operators-Case-Competition/data/severity.xlsx")
library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(patchwork)
library(glmnet)
library(ranger)
library(ggmosaic)
library(embed) 
# Read and Set ------------------------------------------------------------

setwd("~/GitHub/AmazonEmployeeAccess")
train_data <- vroom("train.csv")
test_data <- vroom("test.csv")


# Boxplot -----------------------------------------------------------------


train_data2 <- train_data %>%
  mutate(
    ROLE_ROLLUP_1 = as.factor(ROLE_ROLLUP_1),
    ACTION = as.factor(ACTION)
  )

ggplot(data = train_data2) +
  geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_1), fill = ACTION))

ggplot(data = train_data2) +
  geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_2), fill = ACTION))

# Bake --------------------------------------------------------------------


my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())


# Apply the recipe

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train_data)

logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train_data)

amazon_predictions <- predict(logReg_workflow,
                              new_data=test_data,
                              type=prob)


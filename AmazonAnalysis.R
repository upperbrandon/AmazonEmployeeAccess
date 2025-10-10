# Load Libraries ----------------------------------------------------------
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
test_data  <- vroom("test.csv")

# Ensure ACTION is a factor -----------------------------------------------
train_data <- train_data %>%
  mutate(ACTION = factor(ACTION))

# Exploratory: Mosaic Plots -----------------------------------------------
train_data2 <- train_data %>%
  mutate(
    ROLE_ROLLUP_1 = as.factor(ROLE_ROLLUP_1),
    ROLE_ROLLUP_2 = as.factor(ROLE_ROLLUP_2)
  )

ggplot(train_data2) +
  geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_1), fill = ACTION))

ggplot(train_data2) +
  geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_2), fill = ACTION))

# Recipe ------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors())


# Prep and Bake -----------------------------------------------------------
Prepped <- prep(my_recipe)
baked_train <- bake(Prepped, new_data = train_data)

# Model -------------------------------------------------------------------
logRegModel <- logistic_reg() %>%
  set_engine("glm")

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train_data)

# Predict -----------------------------------------------------------------
amazon_predictions <- predict(logReg_workflow,
                              new_data = test_data,
                              type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

# Create submission -------------------------------------------------------
kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(amazon_predictions)

vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")

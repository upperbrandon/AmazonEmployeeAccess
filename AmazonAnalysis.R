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



# Day 2 -------------------------------------------------------------------

# Recipe ------------------------------------------------------------------
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_embed(all_nominal_predictors(), outcome = vars(ACTION)) %>%  # <-- target encoding
  step_zv(all_predictors())  # remove zero-variance columns

# Model -------------------------------------------------------------------
logRegModel <- logistic_reg(
  penalty = tune(),  # lambda
  mixture = tune()   # alpha
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

logReg_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel)

set.seed(123)
amazon_folds <- vfold_cv(train_data, v = 5, strata = ACTION)

lambda_grid <- grid_regular(
  penalty(range = c(-4, 0)),  # log10 scale: 1e-4 to 1
  mixture(range = c(0, 1)),
  levels = 10
)


# Workflow ----------------------------------------------------------------

tune_results <- tune_grid(
  logReg_workflow,
  resamples = amazon_folds,
  grid = lambda_grid,
  metrics = metric_set(roc_auc)
)

best_params <- select_best(tune_results, metric =  "roc_auc")

final_wf <- finalize_workflow(logReg_workflow, best_params)

final_fit <- final_wf %>%
  fit(data = train_data)

# Predict -----------------------------------------------------------------
amazon_predictions <- predict(final_fit,
                              new_data = test_data,
                              type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(amazon_predictions)

# Create submission -------------------------------------------------------
vroom_write(kaggle_submission, file = "./TargetEncoded_ElasticNet_Preds.csv", delim = ",")

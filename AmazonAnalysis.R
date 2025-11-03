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
library(tensorflow)
library(themis) 

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
# 
# ggplot(train_data2) +
#   geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_1), fill = ACTION))
# 
# ggplot(train_data2) +
#   geom_mosaic(aes(weight = 1, x = product(ROLE_ROLLUP_2), fill = ACTION))

# Recipe ------------------------------------------------------------------
my_recipe0 <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) 

# The original
my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_embed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors())

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_embed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.9)


# Smote Recipe ------------------------------------------------------------

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_embed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_smote(ACTION, neighbors = 5) %>%  # Adjust 'neighbors' as needed
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold = 0.9)


# Prep and Bake -----------------------------------------------------------
Prepped <- prep(my_recipe)
baked_train <- bake(Prepped, new_data = train_data)


# SVM ---------------------------------------------------------------------



# Radial ------------------------------------------------------------------



svmRadial <- svm_rbf(
  rbf_sigma = tune(),  # kernel width
  cost = tune()        # regularization strength
) %>%
  set_mode("classification") %>%
  set_engine("kernlab", maxiter = 50000)

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

myFolds <- vfold_cv(train_data, v = 2)
# Tune the SVM model ------------------------------------------------------
svm_tuned <- tune_grid(
  svm_wf,
  resamples = myFolds,
  grid = 2,
  metrics = metric_set(accuracy)
)

# Select the best parameters ----------------------------------------------
best_svm <- select_best(svm_tuned, metric = "accuracy")

# Finalize workflow with best parameters ---------------------------------
final_svm_wf <- finalize_workflow(svm_wf, best_svm)

# Fit final model on full training data ----------------------------------
fit_svm <- fit(final_svm_wf, data = train_data)

# Predict on test data ---------------------------------------------------
svm_predictions <- predict(fit_svm, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

# Prepare SVM submission -------------------------------------------------
kaggle_svm_submission <- test_data %>%
  select(id) %>%
  bind_cols(svm_predictions)

vroom_write(
  kaggle_svm_submission,
  file = "./SVM_Radi.csv",
  delim = ","
)


# Poly --------------------------------------------------------------------

svmPoly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

myFolds <- vfold_cv(train_data, v = 2)
# Tune the SVM model ------------------------------------------------------
svm_tuned <- tune_grid(
  svm_wf,
  resamples = myFolds,
  grid = 2,
  metrics = metric_set(accuracy)
)

# Select the best parameters ----------------------------------------------
best_svm <- select_best(svm_tuned, metric = "accuracy")

# Finalize workflow with best parameters ---------------------------------
final_svm_wf <- finalize_workflow(svm_wf, best_svm)

# Fit final model on full training data ----------------------------------
fit_svm <- fit(final_svm_wf, data = train_data)

# Predict on test data ---------------------------------------------------
svm_predictions <- predict(fit_svm, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

# Prepare SVM submission -------------------------------------------------
kaggle_svm_submission <- test_data %>%
  select(id) %>%
  bind_cols(svm_predictions)

vroom_write(
  kaggle_svm_submission,
  file = "./SVM_Poly.csv",
  delim = ","
)

# Linear ------------------------------------------------------------------


svmLinear <- svm_linear(cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab", maxiter = 50000)



svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmLinear)

myFolds <- vfold_cv(train_data, v = 2)
# Tune the SVM model ------------------------------------------------------
svm_tuned <- tune_grid(
  svm_wf,
  resamples = myFolds,
  grid = 2,
  metrics = metric_set(accuracy)
)

# Select the best parameters ----------------------------------------------
best_svm <- select_best(svm_tuned, metric = "accuracy")

# Finalize workflow with best parameters ---------------------------------
final_svm_wf <- finalize_workflow(svm_wf, best_svm)

# Fit final model on full training data ----------------------------------
fit_svm <- fit(final_svm_wf, data = train_data)

# Predict on test data ---------------------------------------------------
svm_predictions <- predict(fit_svm, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

# Prepare SVM submission -------------------------------------------------
kaggle_svm_submission <- test_data %>%
  select(id) %>%
  bind_cols(svm_predictions)

vroom_write(
  kaggle_svm_submission,
  file = "./SVM_Linear.csv",
  delim = ","
)


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



# Random Forest -----------------------------------------------------------

my_mod <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# Smaller tuning grid: only 4 combos total
grid_of_tuning_params <- grid_regular(
  mtry(range = c(2, 6)),   
  min_n(range = c(2, 10)), 
  levels = 2               
)

# Use CV

folds <- vfold_cv(train_data, v = 3, strata = ACTION)

rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

# Tune
CV_results <- rf_workflow %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(roc_auc),
    control = control_grid(save_pred = FALSE, verbose = FALSE)
  )

bestTune <- select_best(CV_results, metric = "roc_auc")

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_data)

# Predict
amazon_predictions <- predict(
  final_wf,
  new_data = test_data,
  type = "prob"
) %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(amazon_predictions)

vroom_write(
  kaggle_submission,
  file = "./RandomForestPreds.csv",
  delim = ","
)


# K nearest neighbors -----------------------------------------------------

library(tidymodels)

knn_model <- nearest_neighbor(neighbors = 5) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

knn_fit <- knn_wf %>%
  fit(data = train_data)

amazon_predictions <- predict(knn_fit, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(amazon_predictions)

vroom_write(
  kaggle_submission,
  file = "./Knearest.csv",
  delim = ","
)


# Bayes -------------------------------------------------------------------


library(tidymodels)
library(naivebayes)
library(discrim)
library(vroom)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

myFolds <- vfold_cv(train_data, v = 3)

nb_tuned <- tune_grid(
  nb_wf,
  resamples = myFolds,
  grid = 2
)

best_nb <- select_best(nb_tuned, metric = "accuracy")

final_nb_wf <- finalize_workflow(nb_wf, best_nb)

fit_nb <- fit(final_nb_wf, data = train_data)

amazon_predictions <- predict(fit_nb, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(amazon_predictions)

vroom_write(
  kaggle_submission,
  file = "./NBayest.csv",
  delim = ","
)

# Neural Nets -------------------------------------------------------------

library(tidymodels)

nn_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

nn_model <- mlp(
  hidden_units = tune(),
  penalty = tune(),   # optional L2 regularization
  epochs = 100
) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

set.seed(123)
nn_folds <- vfold_cv(train_data, v = 3, strata = ACTION)

nn_grid <- grid_regular(
  hidden_units(range = c(1, 10)),
  penalty(range = c(-4, -1)),  
  levels = 5
)

nn_tune_results <- tune_grid(
  nn_wf,
  resamples = nn_folds,
  grid = nn_grid,
  metrics = metric_set(roc_auc, accuracy)
)

best_nn <- select_best(nn_tune_results,metric = "roc_auc")

final_nn_wf <- finalize_workflow(nn_wf, best_nn) %>%
  fit(data = train_data)

nn_predictions <- predict(final_nn_wf, new_data = test_data, type = "prob") %>%
  select(.pred_1) %>%
  rename(ACTION = .pred_1)

kaggle_submission <- test_data %>%
  select(id) %>%
  bind_cols(nn_predictions)

vroom_write(kaggle_submission, file = "./NeuralNetTidymodels.csv", delim = ",")

library(tidymodels)
library(ggplot2)

nn_tune_results %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x = hidden_units, y = mean)) +
  geom_point() +
  labs(
    title = "NN Accuracy vs Hidden Units",
    x = "Hidden Units",
    y = "Mean Accuracy"
  )


library(dplyr)

nn_metrics <- nn_tune_results %>%
  collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  mutate(mean = as.numeric(mean),
         hidden_units = as.numeric(hidden_units))


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

Columns <- c(
  "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", 
  "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", 
  "ROLE_FAMILY", "ROLE_CODE"
)

my_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_of(Columns), fn = as.factor) %>%
  step_other(all_of(Columns), threshold = 0.001, other = "other") %>%
  step_dummy(all_of(Columns), one_hot = TRUE)

# Apply the recipe
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train_data)

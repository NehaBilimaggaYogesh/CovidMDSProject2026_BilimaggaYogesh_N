
library(tidyverse)   
library(tidymodels)  
library(themis)      
library(caret)       
library(car)         
library(rpart.plot)  
library(vip)         
library(xgboost)     


au <- read_csv("./data/australia.csv")
uk <- read_csv("./data/united-kingdom.csv")

dim(au)  
dim(uk)  


# STEP 1: Fix encoding and replace blank strings with NA
# UK file has invalid UTF-8 in i14_health_other.
# iconv() converts to valid UTF-8 first, then blank -> NA.

uk <- uk %>%
  mutate(across(where(is.character),
                ~ iconv(., from = "", to = "UTF-8", sub = ""))) %>%
  mutate(across(where(is.character),
                ~ ifelse(trimws(.) == "", NA, .)))
au <- au %>%
  mutate(across(where(is.character),
                ~ iconv(., from = "", to = "UTF-8", sub = ""))) %>%
  mutate(across(where(is.character),
                ~ ifelse(trimws(.) == "", NA, .)))

# STEP 2: Add country identifier

uk <- mutate(uk, country = "UK")
au <- mutate(au, country = "Australia")


# STEP 3: Parse endtime as date object
# as.POSIXct() - base R

uk <- mutate(uk,
             endtime = as.POSIXct(endtime,
                                  format = "%d/%m/%Y %H:%M",
                                  tz = "UTC"))
au <- mutate(au,
             endtime = as.POSIXct(endtime,
                                  format = "%d/%m/%Y %H:%M",
                                  tz = "UTC"))


# STEP 4: Extract week number from qweek string

uk <- mutate(uk,
             week_num = as.integer(
               str_match(qweek, "week (\\d+)")[, 2]))
au <- mutate(au,
             week_num = as.integer(
               str_match(qweek, "week (\\d+)")[, 2]))


# STEP 5: Harmonise location column
# AU uses 'state', UK uses 'region'


uk <- mutate(uk, location = region)
au <- mutate(au, location = state)


# STEP 6: Harmonise employment status labels

clean_employment <- function(x) {
  ifelse(str_detect(tolower(x), "full time"),
         "Full time employment",
  ifelse(str_detect(tolower(x), "part time"),
         "Part time employment",
  ifelse(str_detect(tolower(x), "student"),
         "Student",
  ifelse(str_detect(tolower(x),
         "unemployed|looking for work"),
         "Unemployed",
  ifelse(str_detect(tolower(x),
         "not working|not looking"),
         "Not working",
  ifelse(str_detect(tolower(x), "retired"),
         "Retired",
  ifelse(is.na(x), NA, "Other")))))))
}
uk <- mutate(uk,
             employment_status = clean_employment(employment_status))
au <- mutate(au,
             employment_status = clean_employment(employment_status))


# STEP 7: Fix AU household_children (71% missing)
# Fill from backup column household_children_resp

au <- mutate(au, household_children = ifelse(
  is.na(household_children),
  as.character(household_children_resp),
  as.character(household_children)
))


# STEP 8: PHQ-4 to numeric (0-12)

phq4_to_num <- function(x) {
  ifelse(x == "Not at all",              0,
  ifelse(x == "Several days",            1,
  ifelse(x == "More than half the days", 2,
  ifelse(x == "Nearly every day",        3, NA))))
}

uk <- mutate(uk,
             PHQ4_1_num = phq4_to_num(PHQ4_1),
             PHQ4_2_num = phq4_to_num(PHQ4_2),
             PHQ4_3_num = phq4_to_num(PHQ4_3),
             PHQ4_4_num = phq4_to_num(PHQ4_4),
             PHQ4_total = PHQ4_1_num + PHQ4_2_num +
               PHQ4_3_num + PHQ4_4_num)
au <- mutate(au,
             PHQ4_1_num = phq4_to_num(PHQ4_1),
             PHQ4_2_num = phq4_to_num(PHQ4_2),
             PHQ4_3_num = phq4_to_num(PHQ4_3),
             PHQ4_4_num = phq4_to_num(PHQ4_4),
             PHQ4_total = PHQ4_1_num + PHQ4_2_num +
               PHQ4_3_num + PHQ4_4_num)


# STEP 9: Government trust to numeric

trust_to_num <- function(x) {
  ifelse(x == "A lot of confidence",         4,
  ifelse(x == "A fair amount of confidence", 3,
  ifelse(x == "Not very much confidence",    2,
  ifelse(x == "No confidence at all",        1, NA))))
}

handling_to_num <- function(x) {
  ifelse(x == "Very badly",     1,
  ifelse(x == "Somewhat badly", 2,
  ifelse(x == "Somewhat well",  3,
  ifelse(x == "Very well",      4, NA))))
}


uk <- mutate(uk,
             gov_trust_num    = trust_to_num(WCRex2),
             gov_handling_num = handling_to_num(WCRex1))
au <- mutate(au,
             gov_trust_num    = trust_to_num(WCRex2),
             gov_handling_num = handling_to_num(WCRex1))


# STEP 10: Willingness to isolate
# i9_health:  isolate if unwell   -- Yes=1, Not sure=0.5, No=0
# i11_health: isolate if directed -- Very willing=4 ... 0

for (df_name in c("uk", "au")) {
  df <- get(df_name)
  df <- mutate(df,
    i9_health_num = ifelse(i9_health == "Yes",      1,
                   ifelse(i9_health == "Not sure",   0.5,
                   ifelse(i9_health == "No",         0, NA))),
    i11_health_num = ifelse(i11_health == "Very willing",              4,
                    ifelse(i11_health == "Somewhat willing",           3,
                    ifelse(i11_health ==
                           "Neither willing nor unwilling",            2,
                    ifelse(i11_health == "Somewhat unwilling",         1,
                    ifelse(i11_health == "Very unwilling",             0,
                    NA)))))
  )
  assign(df_name, df)
}


# STEP 11: Mask wearing outcome 

freq_to_num <- function(x) {
  ifelse(x == "Always",      5,
  ifelse(x == "Frequently",  4,
  ifelse(x == "Sometimes",   3,
  ifelse(x == "Rarely",      2,
  ifelse(x == "Not at all",  1, NA)))))
}

mask_freq_cols <- c("i12_health_1", "i12_health_2",
                    "i12_health_3", "i12_health_4")
mask_num_cols  <- paste0(mask_freq_cols, "_num")

for (df_name in c("uk", "au")) {
  df <- get(df_name)
  df <- mutate(df,
    across(all_of(mask_freq_cols), freq_to_num,
           .names = "{.col}_num"),
    mask_score   = rowMeans(pick(all_of(mask_num_cols)),
                            na.rm = TRUE),
    mask_wearing = ifelse(is.nan(mask_score), NA,
                   ifelse(mask_score >= 4, 1, 0)))
  assign(df_name, df)
}

cat("AU mask wearing rate:",
    round(mean(au$mask_wearing, na.rm = TRUE), 3), "\n")
cat("UK mask wearing rate:",
    round(mean(uk$mask_wearing, na.rm = TRUE), 3), "\n")


# STEP 12: Select harmonised columns
# dplyr:: prefix avoids conflict with carets select()

cols_to_keep <- c(
  "RecordNo", "country", "endtime", "qweek", "week_num",
  "age", "gender", "location",
  "household_size", "household_children",
  "employment_status", "weight",
  "i9_health_num", "i11_health_num",
  "cantril_ladder",
  "PHQ4_1_num", "PHQ4_2_num", "PHQ4_3_num",
  "PHQ4_4_num", "PHQ4_total",
  "WCRex2", "gov_trust_num",
  "WCRex1", "gov_handling_num",
  "mask_score", "mask_wearing"
)

uk_clean <- dplyr::select(uk, any_of(cols_to_keep))
au_clean <- dplyr::select(au, any_of(cols_to_keep))


# STEP 13: Mandate classification
# AU: 1 July 2021 | UK: 24 July 2020

au_clean <- mutate(au_clean,
  mandate_period = ifelse(
    as.Date(endtime) >= as.Date("2021-07-01"),
    "after", "before"))
uk_clean <- mutate(uk_clean,
  mandate_period = ifelse(
    as.Date(endtime) >= as.Date("2020-07-24"),
    "after", "before"))

cat("\n--- Mandate split ---\n")
cat("AU before:", sum(au_clean$mandate_period == "before"),
    "| after:",  sum(au_clean$mandate_period == "after"),  "\n")
cat("UK before:", sum(uk_clean$mandate_period == "before"),
    "| after:",  sum(uk_clean$mandate_period == "after"),  "\n")

cat("\nMask rates by period:\n")
cat("AU BEFORE:", round(mean(
  au_clean$mask_wearing[au_clean$mandate_period == "before"],
  na.rm = TRUE), 3), "\n")
cat("AU AFTER: ", round(mean(
  au_clean$mask_wearing[au_clean$mandate_period == "after"],
  na.rm = TRUE), 3), "\n")
cat("UK BEFORE:", round(mean(
  uk_clean$mask_wearing[uk_clean$mandate_period == "before"],
  na.rm = TRUE), 3), "\n")
cat("UK AFTER: ", round(mean(
  uk_clean$mask_wearing[uk_clean$mandate_period == "after"],
  na.rm = TRUE), 3), "\n")


# STEP 14: Combine and save

combined <- bind_rows(au_clean, uk_clean)
write_csv(au_clean,  "australia_clean.csv")
write_csv(uk_clean,  "united-kingdom_clean.csv")
write_csv(combined,  "combined_clean.csv")

dim(combined)
count(combined, country)
count(combined, country, gender)
count(combined, country, employment_status)


# EXPLORATORY PLOTS

clean_plot_data <- combined %>%
  dplyr::select(country, cantril_ladder, PHQ4_total) %>%
  mutate(cantril_ladder = as.numeric(cantril_ladder)) %>%
  filter(!is.na(cantril_ladder), !is.na(PHQ4_total),
         cantril_ladder >= 0, cantril_ladder <= 10,
         PHQ4_total >= 0,     PHQ4_total <= 12)

p_combined <- ggplot(clean_plot_data,
       aes(x = cantril_ladder, y = PHQ4_total)) +
  geom_point(alpha = 0.05) +
  geom_smooth(method = "lm") +
  labs(title = "Wellbeing and Mental Health Score",
       x     = "Wellbeing Score (Cantril Ladder, 0-10)",
       y     = "PHQ-4 Mental Health Score (0-12)")
print(p_combined)
ggsave("scatter_combined.png", p_combined,
       width = 7, height = 5, dpi = 150)
cat("Saved: scatter_combined.png\n")

p_by_country <- ggplot(clean_plot_data,
       aes(x = cantril_ladder, y = PHQ4_total)) +
  geom_point(alpha = 0.05) +
  geom_smooth(method = "lm") +
  facet_wrap(~ country) +
  labs(title = "Wellbeing and Mental Health Score by Country",
       x     = "Wellbeing Score (Cantril Ladder, 0-10)",
       y     = "PHQ-4 Mental Health Score (0-12)")
print(p_by_country)
ggsave("scatter_by_country.png", p_by_country,
       width = 10, height = 5, dpi = 150)
cat("Saved: scatter_by_country.png\n")


# MODEL VARIABLES

model_vars <- c(
  "mask_wearing",
  "age",
  "gender",
  "location",            
  "employment_status",
  "household_size",
  "cantril_ladder",
  "PHQ4_total",
  "gov_trust_num",       
  "gov_handling_num",    
  "week_num",
  "i9_health_num",       
  "i11_health_num"       
)

# Check rows surviving drop_na per condition
cat("\n--- Rows after drop_na per condition ---\n")
for (country_name in c("Australia", "UK")) {
  df <- if (country_name == "Australia") au_clean else uk_clean
  for (period in c("before", "after")) {
    n <- df %>%
      filter(mandate_period == period) %>%
      dplyr::select(any_of(model_vars)) %>%
      drop_na() %>%
      nrow()
    total <- df %>% filter(mandate_period == period) %>% nrow()
    cat(country_name, period, ":", n, "rows (",
        round(n / total * 100, 1), "% retained)\n")
  }
}


# HELPER FUNCTIONS


prepare_data <- function(df, period) {
  df %>%
    filter(mandate_period == period) %>%
    dplyr::select(any_of(model_vars)) %>%
    drop_na() %>%
    mutate(
      mask_wearing      = as.factor(mask_wearing),
      gender            = as.character(gender),
      location          = as.character(location),
      employment_status = as.character(employment_status),
      cantril_ladder    = as.numeric(cantril_ladder),
      PHQ4_total        = as.numeric(PHQ4_total),
      gov_trust_num     = as.numeric(gov_trust_num),
      gov_handling_num  = as.numeric(gov_handling_num),
      week_num          = as.numeric(week_num),
      i9_health_num     = as.numeric(i9_health_num),
      i11_health_num    = as.numeric(i11_health_num)
    )
}

fit_models_upsampled <- function(train_data) {

  rec <- recipe(mask_wearing ~ ., data = train_data) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_upsample(mask_wearing,
                  over_ratio = 1,
                  seed       = 1905585)

  list(
    lr = workflow() %>%
      add_recipe(rec) %>%
      add_model(logistic_reg() %>%
                  set_engine("glm") %>%
                  set_mode("classification")) %>%
      fit(train_data),

    ct = workflow() %>%
      add_recipe(rec) %>%
      add_model(decision_tree(mode = "classification") %>%
                  set_engine("rpart")) %>%
      fit(train_data),

    rf = workflow() %>%
      add_recipe(rec) %>%
      add_model(rand_forest(mode = "classification") %>%
                  set_engine("ranger",
                             importance = "permutation")) %>%
      fit(train_data),

    xgb = workflow() %>%
      add_recipe(rec) %>%
      add_model(boost_tree(trees = 250) %>%
                  set_engine("xgboost") %>%
                  set_mode("classification")) %>%
      fit(train_data)
  )
}

evaluate_models <- function(models, test_data, country, period) {
  labels <- c(lr  = "Logistic Regression",
              ct  = "Classification Tree",
              rf  = "Random Forest",
              xgb = "XGBoost")
  map_dfr(names(models), function(nm) {
    preds <- predict(models[[nm]], test_data,
                     type = "prob") %>%
      bind_cols(predict(models[[nm]], test_data)) %>%
      bind_cols(test_data)
    tibble(
      Country  = country,
      Period   = period,
      Model    = labels[nm],
      AUC      = round(roc_auc(preds,
                   truth = mask_wearing, .pred_1,
                   event_level = "second")$.estimate, 3),
      Accuracy = round(accuracy(preds,
                   truth = mask_wearing,
                   .pred_class)$.estimate, 3)
    )
  })
}


# FIT BASELINE MODELS ALL 16 CONDITIONS

set.seed(1905585)
cat("\n--- Fitting baseline models (with upsampling) ---\n")


# Australia - before mandate
au_b       <- prepare_data(au_clean, "before")
cat("AU before: n =", nrow(au_b),
    "| mask wearers:", round(mean(au_b$mask_wearing == 1) * 100, 1), "%\n")
au_b_split <- initial_split(au_b, prop = 0.8,
                             strata = mask_wearing)
au_b_train <- training(au_b_split)
au_b_test  <- testing(au_b_split)
au_b_mods  <- fit_models_upsampled(au_b_train)
au_b_res   <- evaluate_models(au_b_mods, au_b_test,
                               "Australia", "Before")
cat("AU before done\n")

# Australia - after mandate
au_a       <- prepare_data(au_clean, "after")
cat("AU after: n =", nrow(au_a),
    "| mask wearers:", round(mean(au_a$mask_wearing == 1) * 100, 1), "%\n")
au_a_split <- initial_split(au_a, prop = 0.8,
                             strata = mask_wearing)
au_a_train <- training(au_a_split)
au_a_test  <- testing(au_a_split)
au_a_mods  <- fit_models_upsampled(au_a_train)
au_a_res   <- evaluate_models(au_a_mods, au_a_test,
                               "Australia", "After")
cat("AU after done\n")

# UK - before mandate
uk_b       <- prepare_data(uk_clean, "before")
cat("UK before: n =", nrow(uk_b),
    "| mask wearers:", round(mean(uk_b$mask_wearing == 1) * 100, 1), "%\n")
uk_b_split <- initial_split(uk_b, prop = 0.8,
                             strata = mask_wearing)
uk_b_train <- training(uk_b_split)
uk_b_test  <- testing(uk_b_split)
uk_b_mods  <- fit_models_upsampled(uk_b_train)
uk_b_res   <- evaluate_models(uk_b_mods, uk_b_test,
                               "UK", "Before")
cat("UK before done\n")

# UK - after mandate
uk_a       <- prepare_data(uk_clean, "after")
cat("UK after: n =", nrow(uk_a),
    "| mask wearers:", round(mean(uk_a$mask_wearing == 1) * 100, 1), "%\n")
uk_a_split <- initial_split(uk_a, prop = 0.8,
                             strata = mask_wearing)
uk_a_train <- training(uk_a_split)
uk_a_test  <- testing(uk_a_split)
uk_a_mods  <- fit_models_upsampled(uk_a_train)
uk_a_res   <- evaluate_models(uk_a_mods, uk_a_test,
                               "UK", "After")
cat("UK after done\n")

# All baseline results
all_baseline <- bind_rows(au_b_res, au_a_res,
                          uk_b_res, uk_a_res)
cat("\n=== BASELINE MODEL RESULTS ===\n")
print(all_baseline)


# RANDOM FOREST DIAGNOSTICS AND SAVE PLOTS

cat("\n--- Confusion matrix (RF, AU before) ---\n")
rf_preds_au_b <- predict(au_b_mods$rf, au_b_test,
                         type = "prob") %>%
  bind_cols(predict(au_b_mods$rf, au_b_test)) %>%
  bind_cols(au_b_test)

rf_preds_au_b %>%
  conf_mat(.pred_class, truth = mask_wearing)

# ROC curve
p_roc <- rf_preds_au_b %>%
  roc_curve(truth = mask_wearing, .pred_1,
            event_level = "second") %>%
  autoplot() +
  ggtitle("ROC Curve - Random Forest, Australia Before Mandate")
print(p_roc)
ggsave("roc_au_before.png", p_roc,
       width = 6, height = 5, dpi = 150)
cat("Saved: roc_au_before.png\n")

# Variable importance -- AU before
p_vip_before <- au_b_mods$rf %>%
  vip(num_features = 10) +
  ggtitle("Variable Importance - Australia Before Mandate") +
  theme_minimal()
print(p_vip_before)
ggsave("vip_au_before.png", p_vip_before,
       width = 7, height = 5, dpi = 150)
cat("Saved: vip_au_before.png\n")

# Variable importance - AU after
p_vip_after <- au_a_mods$rf %>%
  vip(num_features = 10) +
  ggtitle("Variable Importance - Australia After Mandate") +
  theme_minimal()
print(p_vip_after)
ggsave("vip_au_after.png", p_vip_after,
       width = 7, height = 5, dpi = 150)
cat("Saved: vip_au_after.png\n")


# HYPERPARAMETER TUNING

cat("\n=== HYPERPARAMETER TUNING ===\n")

set.seed(1905585)
au_b_folds <- vfold_cv(au_b_train, v = 5,
                        strata = mask_wearing)

# Upsampling recipe for tuning -- same pipeline as baseline
tune_rec <- recipe(mask_wearing ~ ., data = au_b_train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_upsample(mask_wearing, over_ratio = 1, seed = 1905585)

# Tune Random Forest: mtry 
cat("\n--- Tuning Random Forest (mtry) ---\n")

rf_tune_wf <- workflow() %>%
  add_recipe(tune_rec) %>%
  add_model(rand_forest(trees = 250, mtry = tune()) %>%
              set_engine("ranger", importance = "permutation") %>%
              set_mode("classification"))

rf_tune_res <- tune_grid(
  rf_tune_wf,
  resamples = au_b_folds,
  grid      = tibble(mtry = c(2, 3, 4, 5, 6, 7, 8)),
  metrics   = metric_set(roc_auc, accuracy)
)

cat("RF tuning results (AUC by mtry):\n")
print(
  collect_metrics(rf_tune_res) %>%
    filter(.metric == "roc_auc") %>%
    dplyr::select(mtry, mean, std_err) %>%
    arrange(desc(mean))
)

best_rf_params <- select_best(rf_tune_res, metric = "roc_auc")
cat("Best mtry:", best_rf_params$mtry, "\n")

rf_tuned_wf <- workflow() %>%
  add_recipe(tune_rec) %>%
  add_model(rand_forest(trees = 250,
                        mtry  = best_rf_params$mtry) %>%
              set_engine("ranger", importance = "permutation") %>%
              set_mode("classification")) %>%
  fit(au_b_train)

rf_tuned_preds <- predict(rf_tuned_wf, au_b_test,
                          type = "prob") %>%
  bind_cols(predict(rf_tuned_wf, au_b_test)) %>%
  bind_cols(au_b_test)

rf_tuned_auc <- roc_auc(rf_tuned_preds,
                        truth = mask_wearing, .pred_1,
                        event_level = "second")$.estimate

cat("Baseline RF AUC (AU before):",
    filter(au_b_res, Model == "Random Forest")$AUC, "\n")
cat("Tuned    RF AUC (AU before):",
    round(rf_tuned_auc, 3), "\n")

#  Tune XGBoost: learn_rate and tree_depth 
cat("\n--- Tuning XGBoost (learn_rate, tree_depth) ---\n")

xgb_tune_wf <- workflow() %>%
  add_recipe(tune_rec) %>%
  add_model(boost_tree(
    trees      = 250,
    learn_rate = tune(),
    tree_depth = tune()) %>%
      set_engine("xgboost") %>%
      set_mode("classification"))

xgb_grid <- expand.grid(
  learn_rate = c(0.05, 0.10, 0.15, 0.20),
  tree_depth = c(4, 6, 8)
)

xgb_tune_res <- tune_grid(
  xgb_tune_wf,
  resamples = au_b_folds,
  grid      = xgb_grid,
  metrics   = metric_set(roc_auc, accuracy)
)

cat("XGBoost tuning results (top 5 by AUC):\n")
print(
  collect_metrics(xgb_tune_res) %>%
    filter(.metric == "roc_auc") %>%
    dplyr::select(learn_rate, tree_depth, mean, std_err) %>%
    arrange(desc(mean)) %>%
    head(5)
)

best_xgb_params <- select_best(xgb_tune_res, metric = "roc_auc")
cat("Best learn_rate:", best_xgb_params$learn_rate,
    "| Best tree_depth:", best_xgb_params$tree_depth, "\n")

xgb_tuned_wf <- workflow() %>%
  add_recipe(tune_rec) %>%
  add_model(boost_tree(
    trees      = 250,
    learn_rate = best_xgb_params$learn_rate,
    tree_depth = best_xgb_params$tree_depth) %>%
      set_engine("xgboost") %>%
      set_mode("classification")) %>%
  fit(au_b_train)

xgb_tuned_preds <- predict(xgb_tuned_wf, au_b_test,
                            type = "prob") %>%
  bind_cols(predict(xgb_tuned_wf, au_b_test)) %>%
  bind_cols(au_b_test)

xgb_tuned_auc <- roc_auc(xgb_tuned_preds,
                          truth = mask_wearing, .pred_1,
                          event_level = "second")$.estimate

cat("Baseline XGBoost AUC (AU before):",
    filter(au_b_res, Model == "XGBoost")$AUC, "\n")
cat("Tuned    XGBoost AUC (AU before):",
    round(xgb_tuned_auc, 3), "\n")

cat("\n=== TUNING SUMMARY (Australia, Before Mandate) ===\n")

tuning_summary <- tibble(
  Model        = c("Random Forest", "XGBoost"),
  Baseline_AUC = c(
    filter(au_b_res, Model == "Random Forest")$AUC,
    filter(au_b_res, Model == "XGBoost")$AUC
  ),
  Tuned_AUC = c(
    round(rf_tuned_auc,  3),
    round(xgb_tuned_auc, 3)
  ),
  Best_params = c(
    paste("mtry =", best_rf_params$mtry),
    paste("lr =",   best_xgb_params$learn_rate,
          "| depth =", best_xgb_params$tree_depth)
  )
)
print(tuning_summary)

cat("\n--- All plots saved ---\n")
cat("scatter_combined.png\n")
cat("scatter_by_country.png\n")
cat("roc_au_before.png\n")
cat("vip_au_before.png\n")
cat("vip_au_after.png\n")
cat("All saved to:", getwd(), "\n")

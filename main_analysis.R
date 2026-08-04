library(tidyverse)
library(tidymodels)
library(themis)        
library(vip)           
library(xgboost)       
library(ranger)        
library(patchwork)     
library(lme4)          
library(broom.mixed)   

au_clean <- read_csv("australia_clean.csv")
uk_clean <- read_csv("united-kingdom_clean.csv")
combined <- read_csv("combined_clean.csv")

cat("Australia rows:", nrow(au_clean), "\n")
cat("UK rows:", nrow(uk_clean), "\n")
cat("Combined rows:",  nrow(combined), "\n")

# MODEL VARIABLES

model_vars <- c(
  "mask_wearing",
  "age", "gender", "location", "employment_status",
  "household_size", "cantril_ladder", "PHQ4_total",
  "gov_trust_num", "gov_handling_num", "week_num",
  "i9_health_num", "i11_health_num"
)

prepare_data <- function(df, period) {
  df %>%
    filter(mandate_period == period) %>%
    dplyr::select(any_of(model_vars)) %>%
    drop_na() %>%
    mutate(
      mask_wearing      = as.factor(mask_wearing),
      gender            = as.factor(gender),
      location          = as.factor(location),
      employment_status = as.factor(employment_status),
      household_size    = as.factor(household_size),
      cantril_ladder    = as.numeric(cantril_ladder),
      PHQ4_total        = as.numeric(PHQ4_total),
      gov_trust_num     = as.numeric(gov_trust_num),
      gov_handling_num  = as.numeric(gov_handling_num),
      week_num          = as.numeric(week_num),
      i9_health_num     = as.numeric(i9_health_num),
      i11_health_num    = as.numeric(i11_health_num)
    )
}

evaluate_models <- function(models, test_data, country, period) {
  results <- list()
  
  standard_models <- list(
    lr = "Logistic Regression",
    ct = "Classification Tree",
    rf = "Random Forest"
  )
  
  for (nm in names(standard_models)) {
    preds <- predict(models[[nm]], test_data, type = "prob") %>%
      bind_cols(predict(models[[nm]], test_data)) %>%
      bind_cols(test_data)
    results[[nm]] <- tibble(
      Country  = country,
      Period   = period,
      Model    = standard_models[[nm]],
      AUC      = round(roc_auc(preds,
                               truth       = mask_wearing,
                               .pred_1,
                               event_level = "second")$.estimate, 3),
      Accuracy = round(accuracy(preds,
                                truth       = mask_wearing,
                                .pred_class)$.estimate, 3)
    )
  }
  
  # XGBoost evaluated separately
  test_xgb <- test_data %>%
    mutate(
      gender            = as.integer(as.factor(gender)),
      location          = as.integer(as.factor(location)),
      employment_status = as.integer(as.factor(employment_status)),
      household_size    = as.integer(as.factor(household_size))
    )
  
  x_test <- test_xgb %>%
    dplyr::select(all_of(models$xgb_cols)) %>%
    as.matrix()
  
  y_test   <- as.integer(as.character(test_data$mask_wearing))
  dtest    <- xgb.DMatrix(data = x_test, label = y_test)
  xgb_prob <- predict(models$xgb, dtest)
  xgb_pred <- ifelse(xgb_prob >= 0.5, 1, 0)
  
  xgb_auc <- roc_auc(
    tibble(
      truth   = factor(y_test, levels = c("0", "1")),
      .pred_1 = xgb_prob
    ),
    truth       = truth,
    .pred_1,
    event_level = "second"
  )$.estimate
  
  results$xgb <- tibble(
    Country  = country,
    Period   = period,
    Model    = "XGBoost (40 bins)",
    AUC      = round(xgb_auc, 3),
    Accuracy = round(mean(xgb_pred == y_test), 3)
  )
  
  bind_rows(results)
}


au_b <- prepare_data(au_clean, "before")
au_a <- prepare_data(au_clean, "after")
uk_b <- prepare_data(uk_clean, "before")
uk_a <- prepare_data(uk_clean, "after")

cat("\nRow counts after drop_na():\n")
cat("AU before:", nrow(au_b), "\n")
cat("AU after: ", nrow(au_a), "\n")
cat("UK before:", nrow(uk_b), "\n")
cat("UK after: ", nrow(uk_a), "\n")


fit_models_updated <- function(train_data, rf_mtry = 5) {
  
  # Calculate class weight for XGBoost
  n_neg <- sum(train_data$mask_wearing == "0")
  n_pos <- sum(train_data$mask_wearing == "1")
  scale_weight <- n_neg / n_pos
  cat("  XGBoost scale_pos_weight:", round(scale_weight, 3), "\n")
  

  rec_standard <- recipe(mask_wearing ~ ., data = train_data) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_upsample(mask_wearing, over_ratio = 1, seed = 1905585)
  
  # XGBoost
  train_xgb <- train_data %>%
    mutate(
      gender            = as.integer(as.factor(gender)),
      location          = as.integer(as.factor(location)),
      employment_status = as.integer(as.factor(employment_status)),
      household_size    = as.integer(as.factor(household_size)),
      mask_wearing      = as.integer(as.character(mask_wearing))
    )
  
  x_train  <- train_xgb %>%
    dplyr::select(-mask_wearing) %>%
    as.matrix()
  y_train  <- train_xgb$mask_wearing
  dtrain   <- xgb.DMatrix(data = x_train, label = y_train)
  
  xgb_model <- xgb.train(
    params = list(
      objective        = "binary:logistic",
      eval_metric      = "auc",
      max_depth        = 4,
      eta              = 0.1,
      tree_method      = "hist",
      max_bin          = 40,
      scale_pos_weight = scale_weight
    ),
    data    = dtrain,
    nrounds = 250,
    verbose = 0
  )
  
  list(
    lr = workflow() %>%
      add_recipe(rec_standard) %>%
      add_model(
        logistic_reg() %>%
          set_engine("glm") %>%
          set_mode("classification")
      ) %>%
      fit(train_data),
    
    ct = workflow() %>%
      add_recipe(rec_standard) %>%
      add_model(
        decision_tree(mode = "classification") %>%
          set_engine("rpart")
      ) %>%
      fit(train_data),
    
    rf = workflow() %>%
      add_recipe(rec_standard) %>%
      add_model(
        rand_forest(mode  = "classification",
                    trees = 250,
                    mtry  = rf_mtry) %>%
          set_engine("ranger", importance = "permutation")
      ) %>%
      fit(train_data),
    
    # XGBoost stored differently
    xgb      = xgb_model,
    xgb_cols = colnames(x_train)
  )
}


set.seed(1905585)

au_b_split <- initial_split(au_b, prop = 0.8, strata = mask_wearing)
au_b_train <- training(au_b_split)
au_b_test  <- testing(au_b_split)
au_b_mods  <- fit_models_updated(au_b_train)
au_b_res   <- evaluate_models(au_b_mods, au_b_test, "Australia", "Before")
cat("Australia Before done\n")

au_a_split <- initial_split(au_a, prop = 0.8, strata = mask_wearing)
au_a_train <- training(au_a_split)
au_a_test  <- testing(au_a_split)
au_a_mods  <- fit_models_updated(au_a_train)
au_a_res   <- evaluate_models(au_a_mods, au_a_test, "Australia", "After")
cat("Australia After done\n")

uk_b_split <- initial_split(uk_b, prop = 0.8, strata = mask_wearing)
uk_b_train <- training(uk_b_split)
uk_b_test  <- testing(uk_b_split)
uk_b_mods  <- fit_models_updated(uk_b_train)
uk_b_res   <- evaluate_models(uk_b_mods, uk_b_test, "UK", "Before")
cat("UK Before done\n")

uk_a_split <- initial_split(uk_a, prop = 0.8, strata = mask_wearing)
uk_a_train <- training(uk_a_split)
uk_a_test  <- testing(uk_a_split)
uk_a_mods  <- fit_models_updated(uk_a_train)
uk_a_res   <- evaluate_models(uk_a_mods, uk_a_test, "UK", "After")
cat("UK After done\n")

all_results <- bind_rows(au_b_res, au_a_res, uk_b_res, uk_a_res)
cat("\n RESULTS \n")
print(all_results)
write_csv(all_results, "partB_model_results.csv")


partA_results <- tribble(
  ~Country,     ~Period,   ~Model,                 ~AUC_PartA,
  "Australia", "Before",  "Logistic Regression",   0.690,
  "Australia", "Before",  "Classification Tree",   0.645,
  "Australia", "Before",  "Random Forest",         0.742,
  "Australia", "Before",  "XGBoost (40 bins)",     0.513,
  "Australia", "After",   "Logistic Regression",   0.759,
  "Australia", "After",   "Classification Tree",   0.710,
  "Australia", "After",   "Random Forest",         0.817,
  "Australia", "After",   "XGBoost (40 bins)",     0.619,
  "UK",        "Before",  "Logistic Regression",   0.640,
  "UK",        "Before",  "Classification Tree",   0.612,
  "UK",        "Before",  "Random Forest",         0.633,
  "UK",        "Before",  "XGBoost (40 bins)",     0.588,
  "UK",        "After",   "Logistic Regression",   0.734,
  "UK",        "After",   "Classification Tree",   0.660,
  "UK",        "After",   "Random Forest",         0.742,
  "UK",        "After",   "XGBoost (40 bins)",     0.583
)

comparison <- partA_results %>%
  left_join(
    all_results %>%
      rename(AUC_PartB = AUC) %>%
      dplyr::select(Country, Period, Model, AUC_PartB),
    by = c("Country", "Period", "Model")
  ) %>%
  mutate(Change = round(AUC_PartB - AUC_PartA, 3))

cat("\n PART A vs PART B COMPARISON \n")
print(comparison)

write_csv(comparison, "partA_vs_partB_comparison.csv")


# RANDOM FOREST
plot_vip <- function(model, title_text) {
  model$rf %>%
    vip(num_features = 10) +
    ggtitle(title_text) +
    theme_minimal() +
    theme(plot.title = element_text(size = 15, face = "bold"),
          axis.title.x = element_text(size= 20),
          axis.title.y = element_text(size= 20),
          axis.text = element_text(size= 25))
}

p_vip_au_b <- plot_vip(au_b_mods, "AU Before Mandate")
p_vip_au_a <- plot_vip(au_a_mods, "AU After Mandate")
p_vip_uk_b <- plot_vip(uk_b_mods, "UK Before Mandate")
p_vip_uk_a <- plot_vip(uk_a_mods, "UK After Mandate")

combined_vip <- (p_vip_au_b | p_vip_au_a) /
  (p_vip_uk_b | p_vip_uk_a) +
  plot_annotation(
    title    = "Variable Importance — Random Forest",
    subtitle = "Top 10 predictors by permutation importance",
    theme    = theme(plot.title = element_text(size = 14,
                                               face = "bold"))
  )

ggsave("vip_all_conditions.png", combined_vip,
       width = 14, height = 10, dpi = 150)
cat("\nSaved: vip_all_conditions.png\n")

# STABILITY ANALYSIS 

run_stability <- function(data, n_runs = 10000,
                          condition_name, rf_mtry = 5) {
  
  cat("\nStarting stability analysis:", condition_name,
      "— n_runs =", n_runs, "\n")
  
  all_importance <- vector("list", n_runs)
  
  for (i in seq_len(n_runs)) {
    
    set.seed(i)
    # Note: initial stability run used 50% sample
    # Final stability analysis uses 80% sample (see final_stability_analysis.R)
    n_sample <- floor(nrow(data) * 0.5)
    train    <- data[sample(nrow(data), n_sample), ]
    wearers     <- train[train$mask_wearing == "1", ]
    non_wearers <- train[train$mask_wearing == "0", ]
    
    if (nrow(wearers) < nrow(non_wearers)) {
      extra <- wearers[sample(nrow(wearers),
                              nrow(non_wearers) - nrow(wearers),
                              replace = TRUE), ]
      train <- bind_rows(train, extra)
    } else if (nrow(non_wearers) < nrow(wearers)) {
      extra <- non_wearers[sample(nrow(non_wearers),
                                  nrow(wearers) - nrow(non_wearers),
                                  replace = TRUE), ]
      train <- bind_rows(train, extra)
    }
    
    # Convert factors to dummy variables
    train_matrix <- train %>%
      mutate(
        gender            = as.integer(as.factor(gender)),
        location          = as.integer(as.factor(location)),
        employment_status = as.integer(as.factor(employment_status)),
        household_size    = as.integer(as.factor(household_size)),
        mask_wearing      = as.integer(as.character(mask_wearing))
      )
    
    x_train <- train_matrix %>%
      dplyr::select(-mask_wearing) %>%
      as.matrix()
    
    y_train <- train_matrix$mask_wearing
    
    # Fit random forest with only 50 trees 
    rf_fit <- ranger(
      x                = x_train,
      y                = as.factor(y_train),
      num.trees        = 50,          # reduced from 250
      mtry             = rf_mtry,
      importance       = "permutation",
      num.threads      = 1,
      seed             = i
    )
    
    imp_vals <- rf_fit$variable.importance
    all_importance[[i]] <- tibble(
      Variable  = names(imp_vals),
      Importance = imp_vals,
      run       = i,
      condition = condition_name
    )
    
    if (i %% 100 == 0) {
      cat("  Completed", i, "/", n_runs, "runs\n")
    }
  }
  
  stable <- bind_rows(all_importance) %>%
    group_by(Variable, condition) %>%
    summarise(
      median_importance = median(Importance, na.rm = TRUE),
      sd_importance     = sd(Importance,     na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(median_importance))
  
  cat("  Done:", condition_name, "\n")
  return(stable)
}

N_RUNS <- 10000

au_b_stable <- run_stability(au_b, N_RUNS, "AU Before", rf_mtry = 5)
write_csv(au_b_stable, "stable_AU_before.csv")
cat("AU Before saved\n")

au_a_stable <- run_stability(au_a, N_RUNS, "AU After",  rf_mtry = 5)
write_csv(au_a_stable, "stable_AU_after.csv")
cat("AU After saved\n")

uk_b_stable <- run_stability(uk_b, N_RUNS, "UK Before", rf_mtry = 5)
write_csv(uk_b_stable, "stable_UK_before.csv")
cat("UK Before saved\n")

uk_a_stable <- run_stability(uk_a, N_RUNS, "UK After",  rf_mtry = 5)
write_csv(uk_a_stable, "stable_UK_after.csv")
cat("UK After saved\n")

all_stable <- bind_rows(au_b_stable, au_a_stable,
                        uk_b_stable, uk_a_stable)
write_csv(all_stable, "stability_results.csv")
cat("\nAll stability results saved.\n")

top10 <- all_stable %>%
  group_by(condition) %>%
  slice_max(median_importance, n = 10) %>%
  mutate(rank = row_number()) %>%
  ungroup()

cat("\n TOP 10 STABLE PREDICTORS BY CONDITION \n")
print(top10 %>% dplyr::select(condition, rank, Variable,
                              median_importance))

write_csv(top10, "top10_stable_predictors.csv")

# Plot
plot_top10 <- function(condition_data, title_text) {
  condition_data %>%
    slice_max(median_importance, n = 10) %>%
    mutate(Variable = reorder(Variable, median_importance)) %>%
    ggplot(aes(x = Variable, y = median_importance)) +
    geom_col(fill = "#028090", alpha = 0.85) +
    geom_errorbar(aes(ymin = median_importance - sd_importance,
                      ymax = median_importance + sd_importance),
                  width = 0.3, colour = "#0D1B4B") +
    coord_flip() +
    labs(title = title_text,
         x     = NULL,
         y     = "Median Importance (10,000 runs)") +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))
}

p_s_au_b <- plot_top10(au_b_stable, "AU — Before Mandate")
p_s_au_a <- plot_top10(au_a_stable, "AU — After Mandate")
p_s_uk_b <- plot_top10(uk_b_stable, "UK — Before Mandate")
p_s_uk_a <- plot_top10(uk_a_stable, "UK — After Mandate")

combined_stable <- (p_s_au_b | p_s_au_a) /
  (p_s_uk_b | p_s_uk_a) +
  plot_annotation(
    title    = "Top 10 Stable Predictors",
    subtitle = paste0("Median importance across ", N_RUNS,
                      " random forest runs"),
    theme    = theme(plot.title = element_text(size = 14,
                                               face = "bold"))
  )

ggsave("stable_predictors.png", combined_stable,
       width = 14, height = 10, dpi = 150)
cat("Saved: stable_predictors.png\n")


all_top10_vars <- top10 %>%
  group_by(Variable) %>%
  summarise(n_conditions = n(), .groups = "drop") %>%
  arrange(desc(n_conditions))

universal <- all_top10_vars %>% filter(n_conditions == 4)

cat("\n UNIVERSAL PREDICTORS top 10 in all 4 conditions \n")
print(universal)


#  Country specific predictors
au_vars <- top10 %>%
  filter(condition %in% c("AU Before", "AU After")) %>%
  pull(Variable) %>% unique()

uk_vars <- top10 %>%
  filter(condition %in% c("UK Before", "UK After")) %>%
  pull(Variable) %>% unique()

cat("\n AU ONLY PREDICTORS \n")
print(setdiff(au_vars, uk_vars))

cat("\n UK ONLY PREDICTORS \n")
print(setdiff(uk_vars, au_vars))

# Mandate shift 
check_mandate_shift <- function(before_cond, after_cond,
                                country_name) {
  before_vars <- top10 %>%
    filter(condition == before_cond) %>% pull(Variable)
  after_vars <- top10 %>%
    filter(condition == after_cond)  %>% pull(Variable)
  
  cat("\n---", country_name, "---\n")
  cat("Dropped after mandate:",
      paste(setdiff(before_vars, after_vars), collapse = ", "), "\n")
  cat("Emerged after mandate:",
      paste(setdiff(after_vars, before_vars), collapse = ", "), "\n")
  cat("Common both periods:  ",
      paste(intersect(before_vars, after_vars), collapse = ", "), "\n")
}

cat("\n MANDATE SHIFT \n")
check_mandate_shift("AU Before", "AU After", "AUSTRALIA")
check_mandate_shift("UK Before", "UK After", "UK")


prepare_data_mixed <- function(df, period) {
  df %>%
    filter(mandate_period == period) %>%
    dplyr::select(any_of(model_vars)) %>%
    drop_na() %>%
    mutate(
      mask_wearing      = as.numeric(as.character(mask_wearing)),
      gender            = as.character(gender),
      location          = as.character(location),
      employment_status = as.character(employment_status),
      household_size    = as.character(household_size),
      cantril_ladder    = as.numeric(cantril_ladder),
      PHQ4_total        = as.numeric(PHQ4_total),
      gov_trust_num     = as.numeric(gov_trust_num),
      gov_handling_num  = as.numeric(gov_handling_num),
      week_num          = as.numeric(week_num),
      i9_health_num     = as.numeric(i9_health_num),
      i11_health_num    = as.numeric(i11_health_num)
    )
}

au_b_mixed <- prepare_data_mixed(au_clean, "before")
au_a_mixed <- prepare_data_mixed(au_clean, "after")
uk_b_mixed <- prepare_data_mixed(uk_clean, "before")
uk_a_mixed <- prepare_data_mixed(uk_clean, "after")


fit_mixed <- function(data, country, period) {
  cat("\nFitting mixed effects model:", country, period, "\n")
  glmer(
    mask_wearing ~
      age +
      gender +
      employment_status +
      household_size +
      cantril_ladder +
      PHQ4_total +
      gov_trust_num +
      gov_handling_num +
      week_num +
      i9_health_num +
      i11_health_num +
      (1 | location),   
    data    = data,
    family  = binomial,
    control = glmerControl(optimizer = "bobyqa",
                           optCtrl   = list(maxfun = 2e5))
  )
}

me_au_b <- fit_mixed(au_b_mixed, "Australia", "Before")
me_au_a <- fit_mixed(au_a_mixed, "Australia", "After")
me_uk_b <- fit_mixed(uk_b_mixed, "UK",        "Before")
me_uk_a <- fit_mixed(uk_a_mixed, "UK",        "After")

cat("\nAll mixed effects models fitted.\n")


cat("\n MIXED EFFECTS: AUSTRALIA BEFORE \n")
summary(me_au_b)

cat("\n MIXED EFFECTS: AUSTRALIA AFTER \n")
summary(me_au_a)

cat("\n MIXED EFFECTS: UK BEFORE \n")
summary(me_uk_b)

cat("\n MIXED EFFECTS: UK AFTER \n")
summary(me_uk_a)


#  13b.coefficient table 
# odds_ratio > 1 = more likely to wear a mask
# odds_ratio < 1 = less likely to wear a mask
# p.value < 0.05 = statistically significant

tidy_mixed <- function(model, country, period) {
  tidy(model, effects = "fixed", conf.int = TRUE) %>%
    mutate(
      country    = country,
      period     = period,
      odds_ratio = round(exp(estimate), 3),
      OR_lower   = round(exp(conf.low),  3),
      OR_upper   = round(exp(conf.high), 3),
      p.value    = round(p.value, 4)
    ) %>%
    dplyr::select(country, period, term,
                  odds_ratio, OR_lower, OR_upper, p.value)
}

mixed_results <- bind_rows(
  tidy_mixed(me_au_b, "Australia", "Before"),
  tidy_mixed(me_au_a, "Australia", "After"),
  tidy_mixed(me_uk_b, "UK",        "Before"),
  tidy_mixed(me_uk_a, "UK",        "After")
)

cat("\n MIXED EFFECTS ODDS RATIOS \n")
print(mixed_results)

write_csv(mixed_results, "mixed_effects_results.csv")


#  13c. Significant predictors

cat("\n SIGNIFICANT PREDICTORS (p < 0.05) \n")
sig <- mixed_results %>%
  filter(term != "(Intercept)", p.value < 0.05) %>%
  arrange(country, period, p.value)
print(sig)

write_csv(sig, "significant_mixed_predictors.csv")


get_random_effects <- function(model, country, period) {
  ranef(model)$location %>%
    as_tibble(rownames = "location") %>%
    rename(random_intercept = `(Intercept)`) %>%
    mutate(
      country = country,
      period  = period,
      random_intercept = round(random_intercept, 4)
    ) %>%
    arrange(desc(random_intercept))
}

all_re <- bind_rows(
  get_random_effects(me_au_b, "Australia", "Before"),
  get_random_effects(me_au_a, "Australia", "After"),
  get_random_effects(me_uk_b, "UK",        "Before"),
  get_random_effects(me_uk_a, "UK",        "After")
)

cat("\n RANDOM EFFECTS BY LOCATION \n")
print(all_re)

write_csv(all_re, "random_effects_location.csv")


plot_fixed_effects <- function(data, title_text) {
  data %>%
    filter(term != "(Intercept)") %>%
    mutate(
      significant = p.value < 0.05,
      term        = reorder(term, odds_ratio)
    ) %>%
    ggplot(aes(x = term, y = odds_ratio,
               colour = significant)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = OR_lower, ymax = OR_upper),
                  width = 0.3) +
    geom_hline(yintercept = 1, linetype = "dashed",
               colour = "grey50") +
    coord_flip() +
    scale_colour_manual(
      values = c("TRUE"  = "#028090",
                 "FALSE" = "#C0C0C0"),
      labels = c("TRUE"  = "Significant (p < 0.05)",
                 "FALSE" = "Not significant")
    ) +
    scale_y_log10() +
    labs(title    = title_text,
         x        = NULL,
         y        = "Odds Ratio (log scale)",
         colour   = NULL) +
    theme_minimal() +
    theme(legend.position  = "bottom",
          plot.title       = element_text(size = 11,
                                          face = "bold"))
}

p_fe_au_b <- plot_fixed_effects(
  filter(mixed_results, country == "Australia", period == "Before"),
  "AU — Before Mandate"
)
p_fe_au_a <- plot_fixed_effects(
  filter(mixed_results, country == "Australia", period == "After"),
  "AU — After Mandate"
)
p_fe_uk_b <- plot_fixed_effects(
  filter(mixed_results, country == "UK", period == "Before"),
  "UK — Before Mandate"
)
p_fe_uk_a <- plot_fixed_effects(
  filter(mixed_results, country == "UK", period == "After"),
  "UK — After Mandate"
)

combined_fe <- (p_fe_au_b | p_fe_au_a) /
  (p_fe_uk_b | p_fe_uk_a) +
  plot_annotation(
    title    = "Mixed Effects Model — Fixed Effects",
    subtitle = "Location treated as random effect",
    theme    = theme(plot.title = element_text(size = 14,
                                               face = "bold"))
  )

ggsave("mixed_effects_fixed.png", combined_fe,
       width = 14, height = 10, dpi = 150)
cat("Saved: mixed_effects_fixed.png\n")


#  14b. Random effects plot 

plot_re <- function(data, title_text) {
  data %>%
    mutate(
      location  = reorder(location, random_intercept),
      direction = ifelse(random_intercept > 0,
                         "Above average", "Below average")
    ) %>%
    ggplot(aes(x = location, y = random_intercept,
               fill = direction)) +
    geom_col() +
    geom_hline(yintercept = 0, linetype = "dashed") +
    coord_flip() +
    scale_fill_manual(
      values = c("Above average" = "#028090",
                 "Below average" = "#E8A020")
    ) +
    labs(title    = title_text,
         subtitle = "How much each location differs from average",
         x        = NULL,
         y        = "Random Intercept",
         fill     = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

p_re_au_b <- plot_re(
  filter(all_re, country == "Australia", period == "Before"),
  "AU Before — Location Effects"
)
p_re_uk_b <- plot_re(
  filter(all_re, country == "UK", period == "Before"),
  "UK Before — Location Effects"
)

combined_re <- p_re_au_b | p_re_uk_b

ggsave("random_effects_location.png", combined_re,
       width = 14, height = 6, dpi = 150)
cat("Saved: random_effects_location.png\n")


cat("\n")

cat("COMPLETE SUMMARY\n")


cat("\n Model Results (XGBoost max_bin = 40) \n")
print(all_results)


cat("\n Part A vs Part B Comparison \n")
print(comparison %>% dplyr::select(Country, Period, Model,
                                   AUC_PartA, AUC_PartB, Change))

cat("\n Universal Predictors all 4 conditions \n")
print(universal$Variable)

cat("\n Significant Mixed Effects Predictors \n")
print(sig %>% dplyr::select(country, period, term,
                            odds_ratio, p.value))

cat("\n Files saved \n")
cat("  partB_model_results.csv\n")
cat("  partA_vs_partB_comparison.csv\n")
cat("  stability_results.csv\n")
cat("  top10_stable_predictors.csv\n")
cat("  mixed_effects_results.csv\n")
cat("  significant_mixed_predictors.csv\n")
cat("  random_effects_location.csv\n")

cat("  vip_all_conditions.png\n")
cat("  stable_predictors.png\n")
cat("  mixed_effects_fixed.png\n")
cat("  random_effects_location.png\n")



library(tidyverse)
library(ranger)
library(patchwork)
library(ROCR)

au_clean <- read_csv("australia_clean.csv")
cat("Australia rows:", nrow(au_clean), "\n")

# SECTION 3: MODEL VARIABLES AND PREPARE DATA

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

# Only AU Before 
au_b <- prepare_data(au_clean, "before")
cat("AU Before rows:", nrow(au_b), "\n")

# SECTION 4: CORE EXPERIMENT FUNCTION

run_experiment <- function(data, n_runs, n_trees,
                           sample_pct, rf_mtry = 5,
                           label = "") {
  
  cat("\n--- Running:", label,
      "| runs =", n_runs,
      "| trees =", n_trees,
      "| sample =", paste0(sample_pct * 100, "%"), "---\n")
  
  start_time     <- Sys.time()
  all_importance <- vector("list", n_runs)
  all_auc        <- numeric(n_runs)
  
  for (i in seq_len(n_runs)) {
    
    set.seed(i)
    
    # Step 1: Sample the data
    n_sample <- floor(nrow(data) * sample_pct)
    train    <- data[sample(nrow(data), n_sample), ]
    
    # Step 2: Upsample minority class manually
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
    
    # Step 3: Make a test set
    set.seed(i + 99999)
    test_idx <- sample(nrow(data), floor(nrow(data) * 0.2))
    test     <- data[test_idx, ]
    
    # Step 4: Convert factors to integers
    convert_cols <- function(df) {
      df %>% mutate(
        gender            = as.integer(as.factor(gender)),
        location          = as.integer(as.factor(location)),
        employment_status = as.integer(as.factor(employment_status)),
        household_size    = as.integer(as.factor(household_size)),
        mask_wearing      = as.integer(as.character(mask_wearing))
      )
    }
    
    train_m <- convert_cols(train)
    test_m  <- convert_cols(test)
    
    x_train <- train_m %>%
      dplyr::select(-mask_wearing) %>%
      as.matrix()
    y_train <- as.factor(train_m$mask_wearing)
    
    x_test  <- test_m %>%
      dplyr::select(-mask_wearing) %>%
      as.matrix()
    y_test  <- test_m$mask_wearing
    
    # Step 5: Fit random forest
    rf_fit <- ranger(
      x           = x_train,
      y           = y_train,
      num.trees   = n_trees,
      mtry        = rf_mtry,
      importance  = "permutation",
      probability = TRUE,
      num.threads = 1,
      seed        = i
    )
    
    # Step 6: Calculate AUC
    preds    <- predict(rf_fit, x_test)$predictions[, 2]
    y_binary <- as.integer(y_test)
    
    auc_val <- tryCatch({
      pred_obj <- ROCR::prediction(preds, y_binary)
      ROCR::performance(pred_obj, "auc")@y.values[[1]]
    }, error = function(e) NA)
    
    all_auc[i] <- auc_val
    
    # Step 7: Extract variable importance
    imp_vals <- rf_fit$variable.importance
    all_importance[[i]] <- tibble(
      Variable   = names(imp_vals),
      Importance = imp_vals,
      run        = i
    )
    
    if (i %% 500 == 0) {
      cat("  Completed", i, "/", n_runs, "runs\n")
    }
  }
  
  end_time  <- Sys.time()
  time_mins <- as.numeric(difftime(end_time, start_time,
                                   units = "mins"))
  
  median_auc <- round(median(all_auc, na.rm = TRUE), 3)
  
  top10 <- bind_rows(all_importance) %>%
    group_by(Variable) %>%
    summarise(
      median_importance = median(Importance, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(median_importance)) %>%
    slice_head(n = 10) %>%
    pull(Variable)
  
  cat("  Done — AUC:", median_auc,
      "| Time:", round(time_mins, 2), "mins\n")
  
  list(
    label      = label,
    n_runs     = n_runs,
    n_trees    = n_trees,
    sample_pct = paste0(sample_pct * 100, "%"),
    median_auc = median_auc,
    time_mins  = round(time_mins, 2),
    top10      = top10
  )
}

# SECTION 5: EXPERIMENT 
# Fixed: runs = 1000, sample = 50%
# Trying with 5, 10, 15, 20, 30, 50 trees

cat("EXPERIMENT 1 UPDATED: Varying trees downwards\n")
cat("Fixed: runs = 1000, sample = 50%\n")

exp1_5  <- run_experiment(au_b, n_runs = 1000, n_trees = 5,
                          sample_pct = 0.5,
                          label = "Exp1: 5 trees")

exp1_10 <- run_experiment(au_b, n_runs = 1000, n_trees = 10,
                          sample_pct = 0.5,
                          label = "Exp1: 10 trees")

exp1_15 <- run_experiment(au_b, n_runs = 1000, n_trees = 15,
                          sample_pct = 0.5,
                          label = "Exp1: 15 trees")

exp1_20 <- run_experiment(au_b, n_runs = 1000, n_trees = 20,
                          sample_pct = 0.5,
                          label = "Exp1: 20 trees")

exp1_30 <- run_experiment(au_b, n_runs = 1000, n_trees = 30,
                          sample_pct = 0.5,
                          label = "Exp1: 30 trees")

exp1_50 <- run_experiment(au_b, n_runs = 1000, n_trees = 50,
                          sample_pct = 0.5,
                          label = "Exp1: 50 trees")


# SECTION 6: EXPERIMENT 2 
# Fixed: trees = 50, sample = 50%
# Varying with runs = 1000, 5000, 10000

cat("EXPERIMENT 2: Varying number of runs\n")
cat("Fixed: trees = 50, sample = 50%\n")

exp2_1000  <- run_experiment(au_b, n_runs = 1000,  n_trees = 50,
                             sample_pct = 0.5,
                             label = "Exp2: 1000 runs")

exp2_5000  <- run_experiment(au_b, n_runs = 5000,  n_trees = 50,
                             sample_pct = 0.5,
                             label = "Exp2: 5000 runs")

exp2_10000 <- run_experiment(au_b, n_runs = 10000, n_trees = 50,
                             sample_pct = 0.5,
                             label = "Exp2: 10000 runs")


# SECTION 7: EXPERIMENT 3 
# Fixed: trees = 50, runs = 1000
# Varying with sample = 50%, 65%, 80%

cat("EXPERIMENT 3: Varying sample percentage\n")
cat("Fixed: trees = 50, runs = 1000\n")

exp3_50 <- run_experiment(au_b, n_runs = 1000, n_trees = 50,
                          sample_pct = 0.50,
                          label = "Exp3: 50% sample")

exp3_65 <- run_experiment(au_b, n_runs = 1000, n_trees = 50,
                          sample_pct = 0.65,
                          label = "Exp3: 65% sample")

exp3_80 <- run_experiment(au_b, n_runs = 1000, n_trees = 50,
                          sample_pct = 0.80,
                          label = "Exp3: 80% sample")


# SECTION 8: RESULTS TABLES

# Experiment 1 results
exp1_results <- tribble(
  ~Experiment,       ~What_Changed,     ~N_Runs, ~N_Trees,
  ~Sample,           ~Median_AUC,       ~Time_Mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_5$n_runs,     exp1_5$n_trees,
  exp1_5$sample_pct,    exp1_5$median_auc,    exp1_5$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_10$n_runs,    exp1_10$n_trees,
  exp1_10$sample_pct,   exp1_10$median_auc,   exp1_10$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_15$n_runs,    exp1_15$n_trees,
  exp1_15$sample_pct,   exp1_15$median_auc,   exp1_15$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_20$n_runs,    exp1_20$n_trees,
  exp1_20$sample_pct,   exp1_20$median_auc,   exp1_20$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_30$n_runs,    exp1_30$n_trees,
  exp1_30$sample_pct,   exp1_30$median_auc,   exp1_30$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_50$n_runs,    exp1_50$n_trees,
  exp1_50$sample_pct,   exp1_50$median_auc,   exp1_50$time_mins
)

# Experiments 2 and 3 results
exp2_exp3_results <- tribble(
  ~Experiment,       ~What_Changed,      ~N_Runs,  ~N_Trees,
  ~Sample,           ~Median_AUC,        ~Time_Mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_1000$n_runs,  exp2_1000$n_trees,
  exp2_1000$sample_pct,  exp2_1000$median_auc,  exp2_1000$time_mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_5000$n_runs,  exp2_5000$n_trees,
  exp2_5000$sample_pct,  exp2_5000$median_auc,  exp2_5000$time_mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_10000$n_runs, exp2_10000$n_trees,
  exp2_10000$sample_pct, exp2_10000$median_auc, exp2_10000$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_50$n_runs,    exp3_50$n_trees,
  exp3_50$sample_pct,    exp3_50$median_auc,    exp3_50$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_65$n_runs,    exp3_65$n_trees,
  exp3_65$sample_pct,    exp3_65$median_auc,    exp3_65$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_80$n_runs,    exp3_80$n_trees,
  exp3_80$sample_pct,    exp3_80$median_auc,    exp3_80$time_mins
)

# Combined full table
all_results <- bind_rows(exp1_results, exp2_exp3_results)

cat("FULL RESULTS TABLE\n")
print(all_results)

write_csv(exp1_results,    "exp1_trees_updated.csv")
write_csv(exp2_exp3_results, "exp2_exp3_results.csv")
write_csv(all_results,     "stability_experiment2_results.csv")
cat("\nSaved: stability_experiment2_results.csv\n")

# SECTION 9: TOP 10 PREDICTOR COMPARISON

cat("TOP 10 PREDICTOR COMPARISON\n")

cat("\nExperiment 1 UPDATED - Varying trees downward:\n")
cat("   5 trees: ", paste(exp1_5$top10,  collapse = ", "), "\n")
cat("  10 trees: ", paste(exp1_10$top10, collapse = ", "), "\n")
cat("  15 trees: ", paste(exp1_15$top10, collapse = ", "), "\n")
cat("  20 trees: ", paste(exp1_20$top10, collapse = ", "), "\n")
cat("  30 trees: ", paste(exp1_30$top10, collapse = ", "), "\n")
cat("  50 trees: ", paste(exp1_50$top10, collapse = ", "), "\n")

cat("\nExperiment 2 - Varying runs (trees=50, sample=50%):\n")
cat("  1000 runs: ", paste(exp2_1000$top10,  collapse = ", "), "\n")
cat("  5000 runs: ", paste(exp2_5000$top10,  collapse = ", "), "\n")
cat(" 10000 runs: ", paste(exp2_10000$top10, collapse = ", "), "\n")

cat("\nExperiment 3 - Varying sample (trees=50, runs=1000):\n")
cat("  50% sample: ", paste(exp3_50$top10, collapse = ", "), "\n")
cat("  65% sample: ", paste(exp3_65$top10, collapse = ", "), "\n")
cat("  80% sample: ", paste(exp3_80$top10, collapse = ", "), "\n")

# Save top 10 comparison
top10_comparison <- tribble(
  ~Experiment,        ~Setting,      ~Top10_Predictors,
  
  "1 - Vary trees",   "5 trees",
  paste(exp1_5$top10,  collapse = ", "),
  
  "1 - Vary trees",   "10 trees",
  paste(exp1_10$top10, collapse = ", "),
  
  "1 - Vary trees",   "15 trees",
  paste(exp1_15$top10, collapse = ", "),
  
  "1 - Vary trees",   "20 trees",
  paste(exp1_20$top10, collapse = ", "),
  
  "1 - Vary trees",   "30 trees",
  paste(exp1_30$top10, collapse = ", "),
  
  "1 - Vary trees",   "50 trees",
  paste(exp1_50$top10, collapse = ", "),
  
  "2 - Vary runs",    "1000 runs",
  paste(exp2_1000$top10,  collapse = ", "),
  
  "2 - Vary runs",    "5000 runs",
  paste(exp2_5000$top10,  collapse = ", "),
  
  "2 - Vary runs",    "10000 runs",
  paste(exp2_10000$top10, collapse = ", "),
  
  "3 - Vary sample",  "50% sample",
  paste(exp3_50$top10, collapse = ", "),
  
  "3 - Vary sample",  "65% sample",
  paste(exp3_65$top10, collapse = ", "),
  
  "3 - Vary sample",  "80% sample",
  paste(exp3_80$top10, collapse = ", ")
)

write_csv(top10_comparison, "stability_experiment2_top10.csv")
cat("Saved: stability_experiment2_top10.csv\n")

# SECTION 10: PLOTS

# Experiment 1 plot
p1_auc <- exp1_results %>%
  ggplot(aes(x = factor(N_Trees,
                        levels = c(5,10,15,20,30,50)),
             y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = Median_AUC),
            vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 1: Vary Trees (downward) - AUC",
       x     = "Number of Trees",
       y     = "Median AUC") +
  theme_minimal()

p1_time <- exp1_results %>%
  ggplot(aes(x = factor(N_Trees,
                        levels = c(5,10,15,20,30,50)),
             y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 1: Vary Trees (downward) - Time",
       x     = "Number of Trees",
       y     = "Time (minutes)") +
  theme_minimal()

# Experiment 2 plot
p2_auc <- exp2_exp3_results %>%
  filter(Experiment == "2 - Vary runs") %>%
  ggplot(aes(x = factor(N_Runs), y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = Median_AUC),
            vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 2: Vary Runs - AUC",
       x     = "Number of Runs",
       y     = "Median AUC") +
  theme_minimal()

p2_time <- exp2_exp3_results %>%
  filter(Experiment == "2 - Vary runs") %>%
  ggplot(aes(x = factor(N_Runs), y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 2: Vary Runs - Time",
       x     = "Number of Runs",
       y     = "Time (minutes)") +
  theme_minimal()

# Experiment 3 plot
p3_auc <- exp2_exp3_results %>%
  filter(Experiment == "3 - Vary sample") %>%
  ggplot(aes(x = Sample, y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = Median_AUC),
            vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 3: Vary Sample - AUC",
       x     = "Sample Percentage",
       y     = "Median AUC") +
  theme_minimal()

p3_time <- exp2_exp3_results %>%
  filter(Experiment == "3 - Vary sample") %>%
  ggplot(aes(x = Sample, y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 3: Vary Sample - Time",
       x     = "Sample Percentage",
       y     = "Time (minutes)") +
  theme_minimal()

# Combine all plots
combined_plots <- (p1_auc | p1_time) /
  (p2_auc | p2_time) /
  (p3_auc | p3_time) +
  plot_annotation(
    title    = "Stability Analysis Parameter Experiment 2",
    subtitle = "AU Before Mandate — Updated Experiment 1 with fewer trees",
    theme    = theme(
      plot.title    = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11)
    )
  )

ggsave("stability_experiment2_plots.png", combined_plots,
       width = 14, height = 16, dpi = 150)
cat("Saved: stability_experiment2_plots.png\n")

# SECTION 11: FINAL SUMMARY

cat("EXPERIMENT 2 COMPLETE\n")

cat("\nExperiment 1 UPDATED - Trees downward (runs=1000, sample=50%):\n")
cat("   5 trees:  AUC =", exp1_5$median_auc,
    "| Time =", exp1_5$time_mins,  "mins\n")
cat("  10 trees:  AUC =", exp1_10$median_auc,
    "| Time =", exp1_10$time_mins, "mins\n")
cat("  15 trees:  AUC =", exp1_15$median_auc,
    "| Time =", exp1_15$time_mins, "mins\n")
cat("  20 trees:  AUC =", exp1_20$median_auc,
    "| Time =", exp1_20$time_mins, "mins\n")
cat("  30 trees:  AUC =", exp1_30$median_auc,
    "| Time =", exp1_30$time_mins, "mins\n")
cat("  50 trees:  AUC =", exp1_50$median_auc,
    "| Time =", exp1_50$time_mins, "mins\n")

cat("\nExperiment 2 - Runs (trees=50, sample=50%):\n")
cat("  1000 runs: AUC =", exp2_1000$median_auc,
    "| Time =", exp2_1000$time_mins,  "mins\n")
cat("  5000 runs: AUC =", exp2_5000$median_auc,
    "| Time =", exp2_5000$time_mins,  "mins\n")
cat(" 10000 runs: AUC =", exp2_10000$median_auc,
    "| Time =", exp2_10000$time_mins, "mins\n")

cat("\nExperiment 3 - Sample (trees=50, runs=1000):\n")
cat("  50% sample: AUC =", exp3_50$median_auc,
    "| Time =", exp3_50$time_mins, "mins\n")
cat("  65% sample: AUC =", exp3_65$median_auc,
    "| Time =", exp3_65$time_mins, "mins\n")
cat("  80% sample: AUC =", exp3_80$median_auc,
    "| Time =", exp3_80$time_mins, "mins\n")

cat("\nFiles saved:\n")
cat("  exp1_trees_updated.csv\n")
cat("  exp2_exp3_results.csv\n")
cat("  stability_experiment2_results.csv\n")
cat("  stability_experiment2_top10.csv\n")
cat("  stability_experiment2_plots.png\n")

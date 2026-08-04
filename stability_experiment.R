library(tidyverse)
library(ranger)
library(patchwork)
library(ROCR)

au_clean <- read_csv("australia_clean.csv")
cat("Australia rows:", nrow(au_clean), "\n")

# MODEL VARIABLES AND PREPARE DATA

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

# Only AU Before needed for this experiment
au_b <- prepare_data(au_clean, "before")
cat("AU Before rows:", nrow(au_b), "\n")


#EXPERIMENT 
# Changes one parameter at a time
# Records AUC and time taken

run_experiment <- function(data, n_runs, n_trees,
                           sample_pct, rf_mtry = 5,
                           label = "") {
  
  cat("\n Running:", label,
      "| runs =", n_runs,
      "| trees =", n_trees,
      "| sample =", paste0(sample_pct * 100, "%"), "\n")
  
  start_time <- Sys.time()
  
  all_importance <- vector("list", n_runs)
  all_auc        <- numeric(n_runs)
  
  for (i in seq_len(n_runs)) {
    
    set.seed(i)
    
    # Sample the data
    n_sample <- floor(nrow(data) * sample_pct)
    train    <- data[sample(nrow(data), n_sample), ]
    
    # Upsample minority class manually
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
    
    # Making a test set
    set.seed(i + 99999)
    test_idx <- sample(nrow(data), floor(nrow(data) * 0.2))
    test     <- data[test_idx, ]
    
    # Convert factors to integers
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
    
    # Fit random forest
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
    
    # Extract variable importance
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


# RUN ALL THREE EXPERIMENTS

cat("EXPERIMENT 1: Varying number of trees\n")
cat("Fixed: runs = 1000, sample = 50%\n")

exp1_50  <- run_experiment(au_b, n_runs = 1000, n_trees = 50,
                           sample_pct = 0.5,
                           label = "Exp1: 50 trees")

exp1_150 <- run_experiment(au_b, n_runs = 1000, n_trees = 150,
                           sample_pct = 0.5,
                           label = "Exp1: 150 trees")

exp1_250 <- run_experiment(au_b, n_runs = 1000, n_trees = 250,
                           sample_pct = 0.5,
                           label = "Exp1: 250 trees")


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


# RESULTS TABLE

results_table <- tribble(
  ~Experiment,       ~What_Changed,       ~N_Runs,  ~N_Trees,
  ~Sample,           ~Median_AUC,         ~Time_Mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_50$n_runs,    exp1_50$n_trees,
  exp1_50$sample_pct,   exp1_50$median_auc,   exp1_50$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_150$n_runs,   exp1_150$n_trees,
  exp1_150$sample_pct,  exp1_150$median_auc,  exp1_150$time_mins,
  
  "1 - Vary trees",  "Number of trees",
  exp1_250$n_runs,   exp1_250$n_trees,
  exp1_250$sample_pct,  exp1_250$median_auc,  exp1_250$time_mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_1000$n_runs,  exp2_1000$n_trees,
  exp2_1000$sample_pct, exp2_1000$median_auc, exp2_1000$time_mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_5000$n_runs,  exp2_5000$n_trees,
  exp2_5000$sample_pct, exp2_5000$median_auc, exp2_5000$time_mins,
  
  "2 - Vary runs",   "Number of runs",
  exp2_10000$n_runs, exp2_10000$n_trees,
  exp2_10000$sample_pct, exp2_10000$median_auc, exp2_10000$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_50$n_runs,    exp3_50$n_trees,
  exp3_50$sample_pct,   exp3_50$median_auc,   exp3_50$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_65$n_runs,    exp3_65$n_trees,
  exp3_65$sample_pct,   exp3_65$median_auc,   exp3_65$time_mins,
  
  "3 - Vary sample", "Sample percentage",
  exp3_80$n_runs,    exp3_80$n_trees,
  exp3_80$sample_pct,   exp3_80$median_auc,   exp3_80$time_mins
)


cat("FULL RESULTS TABLE\n")
print(results_table)

write_csv(results_table, "stability_parameter_experiment.csv")
cat("\nSaved: stability_parameter_experiment.csv\n")


# TOP 10 PREDICTOR COMPARISON

cat("TOP 10 PREDICTOR COMPARISON\n")

cat("\nExperiment 1 - Varying trees (runs=1000, sample=50%):\n")
cat("  50 trees: ",  paste(exp1_50$top10,  collapse = ", "), "\n")
cat(" 150 trees: ",  paste(exp1_150$top10, collapse = ", "), "\n")
cat(" 250 trees: ",  paste(exp1_250$top10, collapse = ", "), "\n")

cat("\nExperiment 2 - Varying runs (trees=50, sample=50%):\n")
cat("  1000 runs: ", paste(exp2_1000$top10,  collapse = ", "), "\n")
cat("  5000 runs: ", paste(exp2_5000$top10,  collapse = ", "), "\n")
cat(" 10000 runs: ", paste(exp2_10000$top10, collapse = ", "), "\n")

cat("\nExperiment 3 - Varying sample (trees=50, runs=1000):\n")
cat("  50% sample: ", paste(exp3_50$top10, collapse = ", "), "\n")
cat("  65% sample: ", paste(exp3_65$top10, collapse = ", "), "\n")
cat("  80% sample: ", paste(exp3_80$top10, collapse = ", "), "\n")

# Save top 10 predictor comparison to CSV
top10_comparison <- tribble(
  ~Experiment,    ~Setting,       ~Top10_Predictors,
  
  "1 - Vary trees", "50 trees",
  paste(exp1_50$top10,  collapse = ", "),
  
  "1 - Vary trees", "150 trees",
  paste(exp1_150$top10, collapse = ", "),
  
  "1 - Vary trees", "250 trees",
  paste(exp1_250$top10, collapse = ", "),
  
  "2 - Vary runs",  "1000 runs",
  paste(exp2_1000$top10,  collapse = ", "),
  
  "2 - Vary runs",  "5000 runs",
  paste(exp2_5000$top10,  collapse = ", "),
  
  "2 - Vary runs",  "10000 runs",
  paste(exp2_10000$top10, collapse = ", "),
  
  "3 - Vary sample", "50% sample",
  paste(exp3_50$top10, collapse = ", "),
  
  "3 - Vary sample", "65% sample",
  paste(exp3_65$top10, collapse = ", "),
  
  "3 - Vary sample", "80% sample",
  paste(exp3_80$top10, collapse = ", ")
)

write_csv(top10_comparison, "stability_top10_comparison.csv")
cat("Saved: stability_top10_comparison.csv\n")

# PLOTS


p1_auc <- results_table %>%
  filter(Experiment == "1 - Vary trees") %>%
  ggplot(aes(x = factor(N_Trees), y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = Median_AUC), vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 1: Vary Trees - AUC",
       x = "Number of Trees", y = "Median AUC") +
  theme_minimal()

p1_time <- results_table %>%
  filter(Experiment == "1 - Vary trees") %>%
  ggplot(aes(x = factor(N_Trees), y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 1: Vary Trees - Time",
       x = "Number of Trees", y = "Time (minutes)") +
  theme_minimal()

p2_auc <- results_table %>%
  filter(Experiment == "2 - Vary runs") %>%
  ggplot(aes(x = factor(N_Runs), y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = Median_AUC), vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 2: Vary Runs - AUC",
       x = "Number of Runs", y = "Median AUC") +
  theme_minimal()

p2_time <- results_table %>%
  filter(Experiment == "2 - Vary runs") %>%
  ggplot(aes(x = factor(N_Runs), y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 2: Vary Runs - Time",
       x = "Number of Runs", y = "Time (minutes)") +
  theme_minimal()

p3_auc <- results_table %>%
  filter(Experiment == "3 - Vary sample") %>%
  ggplot(aes(x = Sample, y = Median_AUC)) +
  geom_col(fill = "#028090", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = Median_AUC), vjust = -0.5, size = 3.5) +
  ylim(0, 1) +
  labs(title = "Exp 3: Vary Sample - AUC",
       x = "Sample Percentage", y = "Median AUC") +
  theme_minimal()

p3_time <- results_table %>%
  filter(Experiment == "3 - Vary sample") %>%
  ggplot(aes(x = Sample, y = Time_Mins)) +
  geom_col(fill = "#0D1B4B", alpha = 0.85, width = 0.5) +
  geom_text(aes(label = paste0(Time_Mins, "m")),
            vjust = -0.5, size = 3.5) +
  labs(title = "Exp 3: Vary Sample - Time",
       x = "Sample Percentage", y = "Time (minutes)") +
  theme_minimal()

combined_plots <- (p1_auc | p1_time) /
  (p2_auc | p2_time) /
  (p3_auc | p3_time) +
  plot_annotation(
    title    = "Stability Analysis Parameter Experiment",
    subtitle = "AU Before Mandate - AUC and Time for each setting",
    theme    = theme(
      plot.title    = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11)
    )
  )

ggsave("stability_parameter_plots.png", combined_plots,
       width = 12, height = 14, dpi = 150)
cat("Saved: stability_parameter_plots.png\n")


# FINAL SUMMARY

cat("EXPERIMENT COMPLETE\n")

cat("\nExperiment 1 - Trees (runs=1000, sample=50%):\n")
cat("  50 trees:  AUC =", exp1_50$median_auc,
    "| Time =", exp1_50$time_mins,  "mins\n")
cat(" 150 trees:  AUC =", exp1_150$median_auc,
    "| Time =", exp1_150$time_mins, "mins\n")
cat(" 250 trees:  AUC =", exp1_250$median_auc,
    "| Time =", exp1_250$time_mins, "mins\n")

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
cat("  stability_parameter_experiment.csv\n")
cat("  stability_parameter_plots.png\n")


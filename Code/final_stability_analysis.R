library(tidyverse)
library(ranger)
library(patchwork)

au_clean <- read_csv("australia_clean.csv")
uk_clean <- read_csv("united-kingdom_clean.csv")

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

au_b <- prepare_data(au_clean, "before")
au_a <- prepare_data(au_clean, "after")
uk_b <- prepare_data(uk_clean, "before")
uk_a <- prepare_data(uk_clean, "after")

cat("Row counts:\n")
cat("AU before:", nrow(au_b), "\n")
cat("AU after: ", nrow(au_a), "\n")
cat("UK before:", nrow(uk_b), "\n")
cat("UK after: ", nrow(uk_a), "\n")

# STABILITY FUNCTION
# Confirmed settings: 50 trees, 1000 runs, 80% sample

run_stability_final <- function(data, condition_name,
                                rf_mtry = 5) {
  
  # Confirmed settings
  N_RUNS      <- 1000
  N_TREES     <- 50
  SAMPLE_PCT  <- 0.80
  
  cat("\nStarting final stability:", condition_name, "\n")
  cat("Settings: trees =", N_TREES,
      "| runs =", N_RUNS,
      "| sample =", paste0(SAMPLE_PCT * 100, "%"), "\n")
  
  start_time     <- Sys.time()
  all_importance <- vector("list", N_RUNS)
  
  for (i in seq_len(N_RUNS)) {
    
    set.seed(i)
    
    # Step 1: Sample 80% of data
    n_sample <- floor(nrow(data) * SAMPLE_PCT)
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
    
    # Step 3: Convert factors to integers
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
    
    # Step 4: Fit random forest 50 trees
    rf_fit <- ranger(
      x           = x_train,
      y           = as.factor(y_train),
      num.trees   = N_TREES,
      mtry        = rf_mtry,
      importance  = "permutation",
      num.threads = 1,
      seed        = i
    )
    
    # Step 5: Extract importance
    imp_vals <- rf_fit$variable.importance
    all_importance[[i]] <- tibble(
      Variable   = names(imp_vals),
      Importance = imp_vals,
      run        = i,
      condition  = condition_name
    )
    
    if (i %% 100 == 0) {
      cat("  Completed", i, "/", N_RUNS, "runs\n")
    }
  }
  
  end_time  <- Sys.time()
  time_mins <- round(as.numeric(difftime(end_time, start_time,
                                         units = "mins")), 2)
  
  stable <- bind_rows(all_importance) %>%
    group_by(Variable, condition) %>%
    summarise(
      median_importance = median(Importance, na.rm = TRUE),
      sd_importance     = sd(Importance,     na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(median_importance))
  
  cat("  Done:", condition_name,
      "| Time:", time_mins, "mins\n")
  
  return(stable)
}

# RUN ALL FOUR CONDITIONS

au_b_stable <- run_stability_final(au_b, "AU Before")
write_csv(au_b_stable, "final_stable_AU_before.csv")
cat("AU Before saved\n")

au_a_stable <- run_stability_final(au_a, "AU After")
write_csv(au_a_stable, "final_stable_AU_after.csv")
cat("AU After saved\n")

uk_b_stable <- run_stability_final(uk_b, "UK Before")
write_csv(uk_b_stable, "final_stable_UK_before.csv")
cat("UK Before saved\n")

uk_a_stable <- run_stability_final(uk_a, "UK After")
write_csv(uk_a_stable, "final_stable_UK_after.csv")
cat("UK After saved\n")

all_stable <- bind_rows(au_b_stable, au_a_stable,
                        uk_b_stable, uk_a_stable)
write_csv(all_stable, "final_stability_results.csv")
cat("\nAll conditions saved.\n")

# TOP 10 PREDICTORS

top10 <- all_stable %>%
  group_by(condition) %>%
  slice_max(median_importance, n = 10) %>%
  mutate(rank = row_number()) %>%
  ungroup()

cat("\n=== TOP 10 STABLE PREDICTORS BY CONDITION ===\n")
print(top10 %>% dplyr::select(condition, rank,
                              Variable, median_importance))

write_csv(top10, "final_top10_stable_predictors.csv")

# UNIVERSAL PREDICTORS

universal <- top10 %>%
  group_by(Variable) %>%
  summarise(n_conditions = n(), .groups = "drop") %>%
  filter(n_conditions == 4) %>%
  arrange(desc(n_conditions))

cat("\n=== UNIVERSAL PREDICTORS (all 4 conditions) ===\n")
print(universal)

# MANDATE SHIFT

check_mandate_shift <- function(before_cond, after_cond,
                                country_name) {
  before_vars <- top10 %>%
    filter(condition == before_cond) %>% pull(Variable)
  after_vars  <- top10 %>%
    filter(condition == after_cond)  %>% pull(Variable)
  
  cat("\n---", country_name, "---\n")
  cat("Dropped after mandate:",
      paste(setdiff(before_vars, after_vars),
            collapse = ", "), "\n")
  cat("Emerged after mandate:",
      paste(setdiff(after_vars, before_vars),
            collapse = ", "), "\n")
  cat("Common both periods:  ",
      paste(intersect(before_vars, after_vars),
            collapse = ", "), "\n")
}

cat("\n=== MANDATE SHIFT ===\n")
check_mandate_shift("AU Before", "AU After", "AUSTRALIA")
check_mandate_shift("UK Before", "UK After", "UK")

# PLOTS

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
         y     = "Median Importance (1,000 runs)") +
    theme_minimal() +
    theme(plot.title = element_text(size = 11, face = "bold"))
}

p_au_b <- plot_top10(au_b_stable, "AU — Before Mandate")
p_au_a <- plot_top10(au_a_stable, "AU — After Mandate")
p_uk_b <- plot_top10(uk_b_stable, "UK — Before Mandate")
p_uk_a <- plot_top10(uk_a_stable, "UK — After Mandate")

combined_plot <- (p_au_b | p_au_a) /
  (p_uk_b | p_uk_a) +
  plot_annotation(
    title    = "Final Stability Analysis — Top 10 Predictors",
    subtitle = "50 trees, 1000 runs, 80% sample per condition",
    theme    = theme(
      plot.title    = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11)
    )
  )

ggsave("final_stable_predictors.png", combined_plot,
       width = 14, height = 10, dpi = 150)
cat("Saved: final_stable_predictors.png\n")

cat("\n=== FILES SAVED ===\n")
cat("  final_stable_AU_before.csv\n")
cat("  final_stable_AU_after.csv\n")
cat("  final_stable_UK_before.csv\n")
cat("  final_stable_UK_after.csv\n")
cat("  final_stability_results.csv\n")
cat("  final_top10_stable_predictors.csv\n")
cat("  final_stable_predictors.png\n")

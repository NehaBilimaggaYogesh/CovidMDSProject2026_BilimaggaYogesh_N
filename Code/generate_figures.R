library(tidyverse)
library(patchwork)
library(vip)
library(ggplot2)

base_theme <- theme_minimal() +
  theme(
    plot.title   = element_text(size = 15, face = "bold"),
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
    axis.text    = element_text(size = 12),
    legend.text  = element_text(size = 12),
    legend.title = element_text(size = 12)
  )

annotation_theme <- theme(
  plot.title    = element_text(size = 18, face = "bold"),
  plot.subtitle = element_text(size = 14)
)

cat("Shared theme defined.\n")

var_labels <- c(
  "location"          = "State or region",
  "week_num"          = "Survey week",
  "age"               = "Age",
  "PHQ4_total"        = "Mental health score",
  "cantril_ladder"    = "Life satisfaction",
  "household_size"    = "Household size",
  "i11_health_num"    = "Willingness to isolate (directed)",
  "i9_health_num"     = "Willingness to isolate (unwell)",
  "gov_trust_num"     = "Government trust",
  "gov_handling_num"  = "Government handling",
  "employment_status" = "Employment type",
  "gender"            = "Gender"
)

rename_vars <- function(df) {
  df %>%
    mutate(Variable = ifelse(
      Variable %in% names(var_labels),
      var_labels[Variable],
      Variable
    ))
}

cat("Variable labels defined.\n")


cat("\n--- Section 1: Variable Importance Plots ---\n")

plot_vip <- function(model, title_text) {

  # Extract importance from ranger model
  imp <- model$rf$fit$fit$fit$variable.importance
  imp_df <- tibble(
    Variable   = names(imp),
    Importance = imp
  ) %>%
    mutate(Variable = ifelse(
      Variable %in% names(var_labels),
      var_labels[Variable],
      Variable
    )) %>%
    slice_max(Importance, n = 10) %>%
    mutate(Variable = reorder(Variable, Importance))

  ggplot(imp_df, aes(x = Variable, y = Importance)) +
    geom_col(fill = "#028090", alpha = 0.85) +
    coord_flip() +
    labs(title = title_text,
         x     = NULL,
         y     = "Permutation Importance") +
    base_theme
}

if (exists("au_b_mods")) {
  p_vip_au_b <- plot_vip(au_b_mods, "Australia — Before Mandate")
  p_vip_au_a <- plot_vip(au_a_mods, "Australia — After Mandate")
  p_vip_uk_b <- plot_vip(uk_b_mods, "United Kingdom — Before Mandate")
  p_vip_uk_a <- plot_vip(uk_a_mods, "United Kingdom — After Mandate")

  combined_vip <- (p_vip_au_b | p_vip_au_a) /
    (p_vip_uk_b | p_vip_uk_a) +
    plot_annotation(
      title    = "Variable Importance — Random Forest",
      subtitle = "Top 10 predictors by permutation importance",
      theme    = annotation_theme
    )

  ggsave("vip_all_conditions.png", combined_vip,
         width = 18, height = 14, dpi = 150)
  cat("Saved: vip_all_conditions.png\n")

} else {
  cat("Skipping VIP plots — model objects not in environment.\n")
  cat("Run main_analysis.R first then source this script.\n")
}

cat("\n--- Section 2: Stability Analysis Plots ---\n")

plot_top10 <- function(condition_data, title_text) {
  condition_data %>%
    rename_vars() %>%
    slice_max(median_importance, n = 10) %>%
    mutate(Variable = reorder(Variable, median_importance)) %>%
    ggplot(aes(x = Variable, y = median_importance)) +
    geom_col(fill = "#028090", alpha = 0.85) +
    geom_errorbar(
      aes(ymin = median_importance - sd_importance,
          ymax = median_importance + sd_importance),
      width = 0.3, colour = "#0D1B4B"
    ) +
    coord_flip() +
    labs(title = title_text,
         x     = NULL,
         y     = "Median Importance (1,000 runs)") +
    base_theme
}

if (file.exists("final_stability_results.csv")) {

  final_stable <- read_csv("final_stability_results.csv",
                            show_col_types = FALSE)

  au_b_stable <- filter(final_stable, condition == "AU Before")
  au_a_stable <- filter(final_stable, condition == "AU After")
  uk_b_stable <- filter(final_stable, condition == "UK Before")
  uk_a_stable <- filter(final_stable, condition == "UK After")

  p_au_b <- plot_top10(au_b_stable, "Australia — Before Mandate")
  p_au_a <- plot_top10(au_a_stable, "Australia — After Mandate")
  p_uk_b <- plot_top10(uk_b_stable, "United Kingdom — Before Mandate")
  p_uk_a <- plot_top10(uk_a_stable, "United Kingdom — After Mandate")

  # Save individual plots — big and clear for report
  ggsave("stable_AU_before.png", p_au_b,
         width = 12, height = 8, dpi = 150)
  ggsave("stable_AU_after.png",  p_au_a,
         width = 12, height = 8, dpi = 150)
  ggsave("stable_UK_before.png", p_uk_b,
         width = 12, height = 8, dpi = 150)
  ggsave("stable_UK_after.png",  p_uk_a,
         width = 12, height = 8, dpi = 150)
  cat("Saved: 4 individual stability plots\n")

  # Also save combined version for reference
  combined_plot <- (p_au_b | p_au_a) /
    (p_uk_b | p_uk_a) +
    plot_annotation(
      title    = "Final Stability Analysis — Top 10 Predictors",
      subtitle = "50 trees, 1,000 runs, 80% sample per condition",
      theme    = annotation_theme
    )

  ggsave("final_stable_predictors.png", combined_plot,
         width = 18, height = 14, dpi = 150)
  cat("Saved: final_stable_predictors.png\n")

} else {
  cat("Skipping stability plots — final_stability_results.csv not found.\n")
  cat("Run final_stability_analysis.R first.\n")
}


cat("\n--- Section 3: Mixed Effects Plots ---\n")

plot_fixed_effects <- function(data, title_text) {
  data %>%
    filter(term != "(Intercept)") %>%
    mutate(
      # Rename terms to plain English
      term = case_match(term,
                        "i11_health_num"    ~ "Willingness to isolate (directed)",
                        "i9_health_num"     ~ "Willingness to isolate (unwell)",
                        "gov_trust_num"     ~ "Government trust",
                        "gov_handling_num"  ~ "Government handling",
                        "cantril_ladder"    ~ "Life satisfaction",
                        "PHQ4_total"        ~ "Mental health score",
                        "household_size"    ~ "Household size",
                        "employment_status" ~ "Employment type",
                        "week_num"          ~ "Survey week",
                        "genderMale"        ~ "Gender (Male)",
                        "age"               ~ "Age",
                        .default            = term
      ),
      significant = p.value < 0.05,
      term        = reorder(term, odds_ratio)
    ) %>%
    ggplot(aes(x = term, y = odds_ratio,
               colour = significant)) +
    geom_point(size = 4) +
    geom_errorbar(
      aes(ymin = OR_lower, ymax = OR_upper),
      width = 0.3, linewidth = 0.8
    ) +
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
    labs(title  = title_text,
         x      = NULL,
         y      = "Odds Ratio (log scale)",
         colour = NULL) +
    base_theme +
    theme(legend.position = "bottom")
}

if (file.exists("mixed_effects_results.csv")) {

  mixed_results <- read_csv("mixed_effects_results.csv",
                             show_col_types = FALSE)

  p_fe_au_b <- plot_fixed_effects(
    filter(mixed_results, country == "Australia", period == "Before"),
    "Australia — Before Mandate"
  )
  p_fe_au_a <- plot_fixed_effects(
    filter(mixed_results, country == "Australia", period == "After"),
    "Australia — After Mandate"
  )
  p_fe_uk_b <- plot_fixed_effects(
    filter(mixed_results, country == "UK", period == "Before"),
    "United Kingdom — Before Mandate"
  )
  p_fe_uk_a <- plot_fixed_effects(
    filter(mixed_results, country == "UK", period == "After"),
    "United Kingdom — After Mandate"
  )

  # Save individual plots — big and clear for report
  ggsave("mixed_AU_before.png", p_fe_au_b,
         width = 12, height = 8, dpi = 150)
  ggsave("mixed_AU_after.png",  p_fe_au_a,
         width = 12, height = 8, dpi = 150)
  ggsave("mixed_UK_before.png", p_fe_uk_b,
         width = 12, height = 8, dpi = 150)
  ggsave("mixed_UK_after.png",  p_fe_uk_a,
         width = 12, height = 8, dpi = 150)
  cat("Saved: 4 individual mixed effects plots\n")

  # Also save combined version for reference
  combined_fe <- (p_fe_au_b | p_fe_au_a) /
    (p_fe_uk_b | p_fe_uk_a) +
    plot_annotation(
      title    = "Mixed Effects Model — Fixed Effects",
      subtitle = "Dots right of 1.0 = more likely to wear a mask. Grey = not significant.",
      theme    = annotation_theme
    )

  ggsave("mixed_effects_fixed.png", combined_fe,
         width = 18, height = 14, dpi = 150)
  cat("Saved: mixed_effects_fixed.png\n")

} else {
  cat("Skipping mixed effects plots — mixed_effects_results.csv not found.\n")
  cat("Run main_analysis.R first.\n")
}

cat("\n--- Section 4: Random Effects Plots ---\n")

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
    base_theme +
    theme(legend.position = "bottom")
}

if (file.exists("random_effects_location.csv")) {

  all_re <- read_csv("random_effects_location.csv",
                     show_col_types = FALSE)

  p_re_au_b <- plot_re(
    filter(all_re, country == "Australia", period == "Before"),
    "Australia Before — Location Effects"
  )
  p_re_uk_b <- plot_re(
    filter(all_re, country == "UK", period == "Before"),
    "United Kingdom Before — Location Effects"
  )

  combined_re <- p_re_au_b | p_re_uk_b

  ggsave("random_effects_location.png", combined_re,
         width = 18, height = 8, dpi = 150)
  cat("Saved: random_effects_location.png\n")

} else {
  cat("Skipping random effects plots random_effects_location.csv not found.\n")
  cat("Run main_analysis.R first.\n")
}

cat("\n=== PLOT GENERATION COMPLETE ===\n")
cat("Files saved (if CSV inputs were available):\n")
cat("  vip_all_conditions.png\n")
cat("  final_stable_predictors.png\n")
cat("  mixed_effects_fixed.png\n")
cat("  random_effects_location.png\n")


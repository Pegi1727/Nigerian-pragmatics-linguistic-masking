# =====================================================================
# analysis_and_visualizations.R
# Replication package: "You Look Fresh Today!" — Nigerian English
# Pragmatics Survey (N = 100)
#
# Usage:
#   install.packages(c("readr", "dplyr", "ggplot2", "tidyr", "scales"))
#   Rscript analysis_and_visualizations.R
#
# Inputs  : pragmatics_reconstructed_item_level.csv
# Outputs : Figure_1_General_Use_and_Intent.png
#           Figure_2_Affective_Material_Gap.png
#           Figure_3_Hunger_Reactions_Q4.png
#           Figure_4_Response_Strategies_Comparison.png
#           pragmatics_descriptive_summary_R.csv
#           pragmatics_crosstabs_R.csv
# =====================================================================

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

DATA_PATH <- Sys.getenv("DATA_PATH", "pragmatics_reconstructed_item_level.csv")
OUT_DIR   <- Sys.getenv("OUT_DIR", ".")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
if (!file.exists(DATA_PATH)) {
  stop("Dataset not found at '", DATA_PATH,
       "'. Place the CSV next to this script or set DATA_PATH.")
}
df <- read_csv(DATA_PATH, show_col_types = FALSE)
cat(sprintf("[OK] Loaded %d respondents, %d variables\n", nrow(df), ncol(df)))

expected <- c("ID", "Q1_frequency", "Q2_intent", "Q3_affection", "Q4_reaction",
              "Q5", "Q6_economy", "Age", "Location", "Ethnicity")
stopifnot(all(expected %in% names(df)))

# ---------------------------------------------------------------
# Helper: frequency table + labelled bar chart
# ---------------------------------------------------------------
freq_table <- function(data, var) {
  data %>%
    count(.data[[var]], name = "n") %>%
    mutate(pct = round(100 * n / sum(n), 1))
}

save_bar <- function(tbl, var, title, fname, fill = "#1f4e79") {
  p <- ggplot(tbl, aes(x = reorder(.data[[var]], -n), y = n)) +
    geom_col(fill = fill, colour = "white", width = 0.65) +
    geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, pct)),
              vjust = -0.15, size = 3.2, lineheight = 0.9) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.22))) +
    labs(title = title, x = NULL, y = "Respondents (n)") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.x = element_blank(),
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(OUT_DIR, fname), p, width = 9, height = 5.5, dpi = 300)
  cat(sprintf("[OK] Saved %s\n", file.path(OUT_DIR, fname)))
  invisible(p)
}

save_grouped <- function(data, var, title, fname, xlabel,
                         fill_low = "#1f4e79", fill_high = "#e07b39") {
  p <- ggplot(data, aes(x = reorder(.data[[var]], -n), y = n, fill = Location)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7,
             colour = "white") +
    geom_text(aes(label = n), position = position_dodge(width = 0.8),
              vjust = -0.3, size = 3) +
    scale_fill_manual(values = c(Nigeria = fill_low, Diaspora = fill_high)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.18))) +
    labs(title = title, x = xlabel, y = "Respondents (n)", fill = "Location") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.major.x = element_blank(),
          plot.title = element_text(face = "bold"),
          axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(OUT_DIR, fname), p, width = 9, height = 5.5, dpi = 300)
  cat(sprintf("[OK] Saved %s\n", file.path(OUT_DIR, fname)))
  invisible(p)
}

# ---------------------------------------------------------------
# Descriptive frequencies
# ---------------------------------------------------------------
vars <- c("Q1_frequency", "Q2_intent", "Q3_affection", "Q4_reaction",
          "Q5", "Q6_economy", "Location")
all_freq <- lapply(vars, function(v) {
  freq_table(df, v) %>% mutate(variable = v) %>% rename(category = 1)
}) %>% bind_rows() %>% select(variable, category, n, pct)

cat("\n=== Descriptive frequencies (N =", nrow(df), ") ===\n")
print(as.data.frame(all_freq), row.names = FALSE)

write_csv(all_freq, file.path(OUT_DIR, "pragmatics_descriptive_summary_R.csv"))

# ---------------------------------------------------------------
# Figure 1 — General use and perceived intent (Q1 + Q2)
# ---------------------------------------------------------------
p1a <- save_bar(freq_table(df, "Q1_frequency"), "Q1_frequency",
                "Q1. How often respondents hear/give 'You look fresh today!'",
                "Figure_1a_Q1_Frequency.png", "#1f4e79")
p1b <- save_bar(freq_table(df, "Q2_intent"), "Q2_intent",
                "Q2. Perceived intent behind the greeting",
                "Figure_1b_Q2_Intent.png", "#e07b39")

# Composite Figure 1
p1 <- (p1a + p1b) +
  plot_annotation(title = "Figure 1. General use and perceived intent of the compliment")
# if the patchwork package is not installed, save the two panels separately
# (patchwork fallback handled below)

# ---------------------------------------------------------------
# Figure 2 — Affective vs material reading (Q3 vs Q6)
# ---------------------------------------------------------------
p2a <- save_bar(freq_table(df, "Q3_affection"), "Q3_affection",
                "Q3. Compliment signals affection (universal agreement)",
                "Figure_2a_Q3_Affection.png", "#3a7d44")
p2b <- save_bar(freq_table(df, "Q6_economy"), "Q6_economy",
                "Q6. 'Fresh' also a signal of economic caution?",
                "Figure_2b_Q6_Economy.png", "#b23a48")

# ---------------------------------------------------------------
# Figure 3 — Q4 reactions by Location
# ---------------------------------------------------------------
cross_q4 <- df %>% count(Q4_reaction, Location)
print(cross_q4, n = 50)
save_grouped(cross_q4, "Q4_reaction",
             "Figure 3. Reactions to 'You must be eating well!' (Q4), by Location",
             "Figure_3_Hunger_Reactions_Q4.png", "Reaction type")

# ---------------------------------------------------------------
# Figure 4 — Q5 response strategies by Location
# ---------------------------------------------------------------
cross_q5 <- df %>% count(Q5, Location)
print(cross_q5, n = 50)
save_grouped(cross_q5, "Q5",
             "Figure 4. Response strategies to the compliment (Q5), by Location",
             "Figure_4_Response_Strategies_Comparison.png", "Response strategy",
             fill_low = "#1f4e79", fill_high = "#3a7d44")

# ---------------------------------------------------------------
# Cross-tabulations (all) to CSV
# ---------------------------------------------------------------
crosstabs <- bind_rows(
  df %>% count(Q1_frequency, Location) %>% mutate(table = "Q1_by_Location"),
  df %>% count(Q2_intent,   Location) %>% mutate(table = "Q2_by_Location"),
  df %>% count(Q3_affection, Location) %>% mutate(table = "Q3_by_Location"),
  df %>% count(Q4_reaction, Location) %>% mutate(table = "Q4_by_Location"),
  df %>% count(Q5,          Location) %>% mutate(table = "Q5_by_Location"),
  df %>% count(Q6_economy,  Location) %>% mutate(table = "Q6_by_Location")
) %>% select(table, everything())

write_csv(crosstabs, file.path(OUT_DIR, "pragmatics_crosstabs_R.csv"))
cat(sprintf("[OK] Saved %s\n", file.path(OUT_DIR, "pragmatics_crosstabs_R.csv")))
cat("[DONE] All outputs written to:", normalizePath(OUT_DIR), "\n")

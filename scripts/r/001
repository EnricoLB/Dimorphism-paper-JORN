set.seed(123)

library(dplyr)
library(here)

# Load data
tabg <- read.csv(here("data", "raven_exports", "tabg.csv"), sep = ";")

# Factor conversion
tabg <- tabg %>%
  mutate(
    Sex = as.factor(Sex),
    Species = as.factor(Species),
    Individual = as.factor(Individual),
    Context = as.factor(ifelse(Duet == 1, "Duet", "Spontaneous"))
  )

# Replace Inf with NA
tabg <- tabg %>%
  mutate(across(where(is.numeric), ~ ifelse(is.infinite(.), NA, .)))

# Select acoustic variables
acoustic_cols <- tabg %>%
  select(where(is.numeric)) %>%
  select(-any_of(c("Duet", "Selection", "Channel"))) %>%
  select_if(~ var(., na.rm = TRUE) != 0) %>%
  names()

# Remove rows with NA
tabg <- tabg %>%
  filter(complete.cases(select(., all_of(acoustic_cols))))

# PCA
acoustic_data <- tabg %>% select(all_of(acoustic_cols))

pca_result <- prcomp(acoustic_data, center = TRUE, scale. = TRUE)

# Save outputs
saveRDS(pca_result, here("results", "pca_global.rds"))
write.csv(tabg, here("results", "clean_data.csv"), row.names = FALSE)

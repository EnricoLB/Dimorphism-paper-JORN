set.seed(123)

library(purrr)
library(lmerTest)
library(here)

tabg <- read.csv(here("results", "clean_data.csv"))

results <- tabg %>%
  split(.$Species) %>%
  map(function(df) {

    acoustic_cols <- df %>%
      select(where(is.numeric)) %>%
      select(-any_of(c("Duet", "Selection", "Channel"))) %>%
      names()

    pca <- prcomp(df[, acoustic_cols], center = TRUE, scale. = TRUE)

    df$PC1 <- pca$x[,1]
    df$PC2 <- pca$x[,2]
    df$PC3 <- pca$x[,3]

    model <- lmer(PC1 ~ Sex * Context + (1|Individual), data = df)

    list(pca = pca, model = model)
  })

saveRDS(results, here("results", "species_models.rds"))

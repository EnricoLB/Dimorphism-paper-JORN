set.seed(123)

library(lmerTest)
library(here)

tabg <- read.csv(here("results", "clean_data.csv"))

pca <- readRDS(here("results", "pca_global.rds"))

# Add PCs
tabg$PC1 <- pca$x[,1]
tabg$PC2 <- pca$x[,2]
tabg$PC3 <- pca$x[,3]

model_PC1 <- lmer(PC1 ~ Sex * Context + (1 | Individual), data = tabg)

saveRDS(model_PC1, here("results", "lmm_PC1.rds"))
capture.output(summary(model_PC1), file = here("results", "lmm_PC1_summary.txt"))

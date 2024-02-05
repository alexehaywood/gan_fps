library(tidyverse)
library(ggplot2)

setwd('~/Documents/GAN_Paper/Code/gitrepo/Lipidomics_lasso/')

features <- readxl::read_xlsx("List of selected features from CBGS 1.xlsx") %>%
  unlist() %>% unname()

dat_train <- openxlsx::read.xlsx('MetabData.xlsx', sheet = 1)
dat_ext <- openxlsx::read.xlsx('MetabData.xlsx', sheet = 3)
dat_val <- openxlsx::read.xlsx('MetabData.xlsx', sheet = 4)

colnames(dat_train)[2] <- 'label'
colnames(dat_ext)[2] <- 'label'
colnames(dat_val)[2] <- 'label'

dat_train <- dat_train %>% select(-1)
dat_ext <- dat_ext %>% select(-1)
dat_val <- dat_val %>% select(-1)

dat_train <- dat_train %>% select(all_of(c("label", features)))
dat_ext <- dat_ext %>% select(all_of(c("label", features)))
dat_val <- dat_val %>% select(all_of(c("label", features)))

sum(colnames(dat_train) == colnames(dat_ext))
sum(colnames(dat_train) == colnames(dat_val))

unique(dat_train$label)
unique(dat_ext$label)
unique(dat_val$label)
#make 2 class (excluding external data)
dat_train <- dat_train %>% filter(label == 'Formula' | label == 'HM')
dat_val <- dat_val %>% filter(label == 'Formula' | label == 'HM')

check_na <- function(x){ which( sum(is.na(x))>0 )}
dat_train %>% apply(1, check_na)
dat_ext %>% apply(1, check_na)
dat_val %>% apply(1, check_na)

summary(as.factor(dat_train$label))
summary(as.factor(dat_ext$label))
summary(as.factor(dat_val$label))

#make unbalance in training data
set.seed(117)
n_over <- sum(dat_train$label == 'Formula')
dat_train_under <- dat_train %>% filter(label == 'HM') %>% sample_n(floor(n_over*0.6))
dat_train <- dat_train %>% filter(label == 'Formula') %>% add_row(dat_train_under)

summary(as.factor(dat_train$label))
summary(as.factor(dat_val$label))

#change factor values
dat_train$label <- as.factor(dat_train$label)
dat_val$label <- as.factor(dat_val$label)

dat_train$label <- plyr::revalue(dat_train$label, c('HM' = 'case', 'Formula' = 'control'))
dat_val$label <- plyr::revalue(dat_val$label, c('HM' = 'case', 'Formula' = 'control'))


dat_train %>% write.csv('dat_real_combo.csv')
dat_train %>% filter(label == 'control') %>% write.csv('dat_real_class2.csv')
dat_train %>% filter(label == 'case') %>% write.csv('dat_real_class1.csv')
dat_val %>% write.csv('dat_val.csv')
dat_ext %>% write.csv('dat_ext.csv')

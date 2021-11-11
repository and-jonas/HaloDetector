
# ======================================================================================================== -
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Analyze sampled training data and train classifiers
# Last edited: 2021-11-11
# ======================================================================================================== -

rm(list = ls())

.libPaths("C:/Users/anjonas/Documents/R/R-3.6.2/library")
library(data.table)
library(tidyverse)
library(ggsci)
library(prospectr)
library(caret)
library(pryr)
library(RVenn)
library(nls.multstart)
library(segmented)
library(ggbiplot)
library(xgboost)

wd = "Z:/Public/Jonas/001_LesionZoo/MRE"
setwd(wd)

# load auxiliary functions
source("R/model_utils.R")

# ======================================================================================================== -
# Load and pre-process data ----
# ======================================================================================================== -

# get file names (all samples)
fnames_pos <- list.files("train_data_lesions/Positives/Scans/Profiles", full.names = T, pattern = ".csv")
fnames_neg <- list.files("train_data_lesions/Negatives/Scans/Profiles", full.names = T, pattern = ".csv")
fnames <- c(fnames_pos, fnames_neg)

# template
template <- data.table::fread(fnames[1]) %>% as_tibble() %>% dplyr::slice(0)

# calculate means of smoothed profiles
means = list()
for (j in 1:length(fnames)){
  
  print(paste("Processing", j, "/", length(fnames)))
  print(fnames[j])
  
  # load data
  file = fnames[j]
  data <- data.table::fread(file, header = TRUE)
  
  # get file name
  bname <- basename(file) %>% gsub(".csv", "", .)
  
  # get sample label
  label <- ifelse(grepl("Positives", file), "pos", "neg")
  
  # add to data
  data$.id <- bname
  data$label <- label
  
  # to tibble
  data <- data %>% as_tibble()
  
  # use template to complete df if necessary (missing variables for smaller lesions)
  data <- bind_rows(template, data)
  
  # re-arrange
  data <- data %>% dplyr::select(.id, label, everything())
  
  # separate data from sample info
  info <- data[1:2]
  preds <- data[3:length(data)]
  
  # reshape data 
  # each channel profile must have its own row for smoothing
  out <- list()
  for (i in 1:42){
  # for (i in 1:54){
    # select data for channel
    df <- preds[((i-1)*76+1):(i*76)]
    # get channel name
    channel <- strsplit(names(df),"_") %>% lapply("[[", 1) %>% unlist() %>% unique()
    # get color_space name
    color_space <- strsplit(names(df),"_") %>% lapply("[[", 2) %>% unlist() %>% unique()
    # get channel and color_space
    channel_color_space <- paste(channel, color_space, sep = "_")
    # get type 
    type <- strsplit(names(df),"_") %>% lapply("[[", 4) %>% unlist() %>% unique()
    # add this to the data
    names(df) <- strsplit(names(df), "_") %>% lapply("[[", 3) %>% unlist()
    df$channel <- channel
    df$color_space <- color_space
    df$channel_color_space <- channel_color_space
    df$type <- type
    # rearrange columns
    df <- df %>% dplyr::select(channel, color_space, channel_color_space, type, everything())
    # "extrapolate" by simple "extension"
    na_cols <- which(colSums(is.na(df)) > 0)
    non_na_col <- which(colSums(!is.na(df[4:length(df)])) > 0)[1] + 3
    df[,na_cols] <- df[,non_na_col]
    
    # add info
    out[[i]] <- cbind(info, df)
  }
  lw0 <- data.table::rbindlist(out) %>% as_tibble()
  
  if(is.character(lw0$`0`)){
    print("Skipping")
    next
  }  
  
  # apply the moving average
  lw_smth7 <- prospectr::movav(lw0[7:length(lw0)], w = 7)

  # reshape for summarising and plotting
  lw_smth <- lw_smth7 %>% as_tibble() %>% bind_cols(lw0[1:6], .) %>% 
    pivot_longer(., 7:length(.), names_to = "posY", values_to = "value") %>% 
    mutate(posY = as.numeric(posY))
  
  # calculate means over all profiles
  means[[j]] <- lw_smth %>% group_by(.id, label, channel, color_space, posY, type) %>% 
    dplyr::summarise(mean = mean(value),
                     sd = sd(value))
  
}

all_means <- means %>% bind_rows() %>% group_by(label) %>% group_nest() %>% 
  mutate(checker = purrr::map_lgl(data, ~any(is.na(.[.$posY == -45, "mean"])))) %>% 
  filter(checker != TRUE) 

all_means <- all_means%>% 
  unnest(data) %>% 
  dplyr::select(-checker)

# saveRDS(all_means, "train_data_lesions/average_color_profiles.rds")

# ======================================================================================================== -
# Extract "higher-level" features ----
# ======================================================================================================== -

all_means <- readRDS("train_data_lesions/average_color_profiles.rds")

# > Breakpoint model for v_Luv ----
## get data
data_v <- all_means %>% 
  filter(channel == "v", color_space == "Luv", type=="sc") %>% 
  group_by(label, .id)

## fit breakpoint models
data_fits <- data_v %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
  unnest(fit_bp_lm)

bp_pars_v <- data_fits %>% dplyr::select(.id, slp1:slpdiff)
names(bp_pars_v)[2:length(bp_pars_v)] <- paste0(names(bp_pars_v)[2:length(bp_pars_v)], "_v")

# ======================================================================================================== -

# > Breakpoint model for R_RGB ----
## get data
data_R <- all_means %>% 
  filter(channel == "R", color_space == "RGB", type == "sc") %>% 
  group_by(label, .id)

## fit breakpoint models
data_fits <- data_R %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_bp_lm = purrr::map(data, breakpoint_lm)) %>% 
  unnest(fit_bp_lm)

bp_pars_R <- data_fits %>% dplyr::select(.id, slp1:slpdiff)
names(bp_pars_R)[2:length(bp_pars_R)] <- paste0(names(bp_pars_R)[2:length(bp_pars_R)], "_R")

# ======================================================================================================== -

# > Logistic model for H_HSV ----
## get data
data_H <- all_means %>% 
  filter(channel == "H", color_space == "HSV", type == "sc") %>% 
  mutate(posY = posY + 32)

## fit logistic models
data_fits <- data_H %>%
  dplyr::select(.id, label, posY, mean) %>% 
  as.data.frame() %>%
  tidyr::nest(data = c(posY, mean)) %>%
  group_by(.id) %>%
  mutate(fit_log = purrr::map(data,
                              ~ nls_multstart(mean ~ logistic(c, d, b, e, posY = posY),
                                              data = .x,
                                              iter = 750,
                                              start_lower = c(c = -0.5, d = 0.5, e = 20, b = -0.5),
                                              start_upper = c(c = 0.5, d = 1.5, e = 44, b = 0),
                                              convergence_count = 150,
                                              supp_errors = 'Y'))) %>%
  tidyr::gather(met, fit, fit_log:fit_log)

# get model parameters
mod_pars <- data_fits %>%
  #reconverto to wide
  dplyr::filter(met == "fit_log") %>% 
  tidyr::spread(met, fit) %>% 
  mutate(p = purrr::map(fit_log, broom::tidy)) %>% 
  unnest(p) %>% 
  dplyr::select(1:6) %>% 
  tidyr::spread(term, estimate) 

log_pars <- mod_pars %>% dplyr::select(.id, b, c, d, e)

# ======================================================================================================== -
# Prepare data for modelling ----
# ======================================================================================================== -

# reshape data
data_reshape <- all_means %>% 
  # do not require standard deviation
  dplyr::select(-sd) %>% 
  # create new variable
  mutate(var = paste(channel, color_space, posY, type, sep = "_")) %>% 
  # drop old separated variables 
  dplyr::select(-channel, -color_space, -posY, -type) %>% 
  pivot_wider(names_from = "var", values_from = "mean")

drop <- paste(paste0(as.character(seq(-30, 37, by = 2)), "_"), collapse = "|")
names_drop <- grep(drop, names(data_reshape), value = TRUE)

data_mod_red <- data_reshape %>% 
  # increase the distance between profile neighbouring pixels
  dplyr::select(-one_of(names_drop)) %>% 
  # these channels appear to be redundant (perfectly correlated with another channel)
  dplyr::select(-starts_with("L_Lab"), -contains("YCC"))

# add model parameters as predictors
# (join by .id)
d_mod <- data_mod_red %>% 
  full_join(., bp_pars_v, by = ".id") %>% 
  full_join(., bp_pars_R, by = ".id") %>% 
  full_join(., log_pars, by = ".id") %>% 
  dplyr::select(-.id)
saveRDS(d_mod, "train_data_lesions/training_data.rds")
template <- d_mod %>% dplyr::select(-label) %>% dplyr::slice(1)
write_csv(template, "train_data_lesions/template_varnames.csv")

# ======================================================================================================== -
# Model ----
# ======================================================================================================== -

d_mod <- readRDS("train_data_lesions/training_data.rds")

# > pls ----

n_pos <- d_mod %>% filter(label == "pos") %>% nrow()
n_neg <- d_mod %>% filter(label == "neg") %>% nrow()

# train and validate 
indx <- createMultiFolds(d_mod$label, k = 10, times = 5)
ctrl <- caret::trainControl(method = "repeatedcv", 
                            index = indx,
                            classProbs = TRUE,
                            savePredictions = TRUE,
                            verboseIter = TRUE,
                            selectionFunction = "oneSE",
                            allowParallel = TRUE)
tune_length <- 20
plsda <- train(label ~., 
               data = d_mod, 
               preProc = c("center", "scale"),
               method = "pls",
               tuneLength = tune_length, 
               trControl = ctrl,
               returnResamp = "all")
plot(plsda)
imp <- varImp(plsda)
importance <- imp$importance

## SAVE MODEL FOR IMPORT IN PYTHON

MODEL_SAVE_PATH = "Output/Models/pls"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/pls")

# save
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
# dir.create(dirname(model_path), showWarnings=FALSE, recursive=TRUE)
saveRDS(plsda, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

# mod <- readRDS(model_rds_path)

# ======================================================================================================== -

# > random forest ----

#specify model tuning parameters
mtry <- c(3, 5, 7, 9, 12, 15)
min_nodes <- c(2, 3, 5, 8)
tune_grid <- expand.grid(mtry = mtry,
                         splitrule = "gini", # default
                         min.node.size = min_nodes)

rf_ranger <- caret::train(label ~ .,
                          data = d_mod,
                          preProc = c("center", "scale"),
                          method = "ranger",
                          tuneGrid = tune_grid,
                          importance = "permutation",
                          num.trees = 500,
                          trControl = ctrl)
plot(rf_ranger)
imp <- varImp(rf_ranger)
imp$importance

# save
MODEL_SAVE_PATH = "Output/Models/spl/rf_v2.1"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/ranger")
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
# dir.create(dirname(model_path), showWarnings=FALSE, recursive=TRUE)
saveRDS(rf_ranger, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

# ======================================================================================================== -

# > support vector machine ----
## linear
svmLin <- caret::train(label ~., 
                    data = d_mod, 
                    preProc = c("center", "scale"),
                    method = "svmLinear", 
                    trControl = ctrl, 
                    preProcess = c("center","scale"),
                    tuneGrid = expand.grid(C = c(0.01, 0.25, 0.50, 0.1, 2, 4, 8, 16, 32, 
                                                 64, 128, 256)))

# save
MODEL_SAVE_PATH = "Output/Models/spl/svmLin_v2.1"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/kernlab")
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
saveRDS(svmLin, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

## non-linear
svm <- caret::train(label ~., 
                    data = d_mod, 
                    preProc = c("center", "scale"),
                    method = "svmRadial", 
                    trControl = ctrl, 
                    preProcess = c("center","scale"), 
                    tuneLength = 20)

plot(svm)

# save
MODEL_SAVE_PATH = "Output/Models/spl/svmRad_v2.1"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/kernlab")
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
saveRDS(svm, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

# ======================================================================================================== -

# > XGBoost ----

nrounds <- 1000

# note to start nrounds from 200, as smaller learning rates result in errors so
# big with lower starting points that they'll mess the scales
tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

xgb_tune <- caret::train(
  label ~., 
  data = d_mod,
  trControl = ctrl,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)

# save
MODEL_SAVE_PATH = "Output/Models/spl/xgboost_v2.1"
DEP_LIBS = c("C:/Users/anjonas/RLibs/caret", "C:/Users/anjonas/RLibs/xgboost")
model_rds_path = paste(MODEL_SAVE_PATH, ".rds", sep='')
model_dep_path = paste(MODEL_SAVE_PATH, ".dep", sep='')

# save model
saveRDS(xgb_tune, model_rds_path)

# save dependency list
file_conn <- file(model_dep_path)
writeLines(DEP_LIBS, file_conn)
close(file_conn)

# ======================================================================================================== -

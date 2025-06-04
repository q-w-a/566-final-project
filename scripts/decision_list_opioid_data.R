
scripts_path <- here::here("supplementary_code_files/simulations")
# source(paste0(scripts_path, "/regimes.R"))
source(paste0(scripts_path, "/sim_util.R"))

dyn.load(paste0("supplementary_code_files/simulations/dtr", .Platform$dynlib.ext))
source(paste0(scripts_path, "/wsvm.R"))

# estimate decision list 
estimate.regime.dl <- function(y, a, x, stage.x, seed = 1988, max.length)
{
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)
  dtr <- vector("list", n.stage)
  
  set.seed(seed)
  num.folds <- 5L
  fold <- rep_len(0L : (num.folds - 1L), n)
  
  future.y <- double(n)
  
  for (i.stage in n.stage : 1L) {
    current.x <- cbind(x[, which(stage.x <= i.stage), drop = FALSE],
                       a[, seq_len(i.stage - 1L), drop = FALSE],
                       y[, seq_len(i.stage - 1L), drop = FALSE])
    current.a <- a[, i.stage]
    current.y <- y[, i.stage] + future.y
    
    if (ncol(current.x) < 2) {
      current.x <- cbind(current.x, 0)
    }
    
    options <- sort.int(unique(current.a))
    group <- as.integer(factor(current.a, options)) - 1L
    
    var.x <- colMeans(current.x ^ 2) - colMeans(current.x) ^ 2
    var.x[var.x < 1e-8] <- Inf
    scaling <- 1 / var.x
    model <- .Call("R_kernel_train", current.x, current.y,
                   group, length(options), scaling)
    outcomes <- .Call("R_kernel_predict", model, current.x)
    regrets <- .Call("R_get_regrets_from_outcomes", outcomes)
    
    # tuning
    
    simple.regret <- min(colMeans(regrets))
    if (simple.regret > 1e-8) {
      zeta.grid <- simple.regret * c(2, 0.75, 0.3, 0.12, 0.05)
      eta.grid <- simple.regret * n * c(0.3, 0.1, 0.03)
      zeta.choices <- rep(zeta.grid, times = 3L)
      eta.choices <- rep(eta.grid, each = 5L)
      current.fold <- fold[sample.int(n)]
      
      cv.regret <- .Call("R_cv_tune_rule", current.x, regrets,
                         zeta.choices, eta.choices, max.length,
                         current.fold, num.folds)
      min.cv.regret <- min(cv.regret)
      
      if (min.cv.regret < simple.regret - 1e-8) {
        index <- which(cv.regret - min.cv.regret - 1e-8 <= 0)[1L]
        zeta.selected <- zeta.choices[index]
        eta.selected <- eta.choices[index]
      } else {
        zeta.selected <- simple.regret * n
        eta.selected <- simple.regret * n
        max.length <- 1L
      }
    } else {
      zeta.selected <- simple.regret * n
      eta.selected <- simple.regret * n
      max.length <- 1L
    }
    
    action <- integer(n)
    dtr[[i.stage]] <- .Call("R_find_rule", current.x, regrets,
                            zeta.selected, eta.selected, max.length, action)
    attr(dtr[[i.stage]], "options") <- options
    
    action <- action + 1L
    future.y <- outcomes[cbind(1L : n, action)]
  }
  
  return(list(dtr=dtr, value=mean(future.y)))
}

# print estimated decision list 
print.regime <- function(dtr, variables)
{
  for (i.stage in seq_along(dtr)) {
    cat(sprintf("Stage %d:\n", i.stage))
    d <- dtr[[i.stage]]
    op1 <- ifelse(substring(d[, "type"], 1, 1) == "L", "<=", ">")
    op2 <- ifelse(substring(d[, "type"], 2, 2) == "L", "<=", ">")
    var1 <- variables[d[, "j1"]]
    var2 <- variables[d[, "j2"]]
    term1 <- ifelse(is.finite(d[, "c1"]), paste(var1, op1, d[, "c1"]), "")
    term2 <- ifelse(is.finite(d[, "c2"]), paste(var2, op2, d[, "c2"]), "")
    cond <- ifelse(nchar(term1) > 0 & nchar(term2) > 0,
                   paste(term1, "and", term2), paste0(term1, term2))
    act <- attr(d, "options")[d[, "a"] + 1]
    cat(paste0("if ", cond, " then ", act, "\n"))
  }
  invisible(NULL)
}

# get the actions assigned by the given decision list 
apply.regime.dl <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x) || ncol(current.x) < 2) {
    current.x <- cbind(current.x, 0)
  }
  
  action <- .Call("R_apply_rule", current.dtr, current.x) + 1L
  attr(current.dtr, "options")[action]
}

# load data 
opioid_dat <- readRDS("data/final_data.RDS")

# formatting input 
opioid_dat <- opioid_dat %>%
  mutate(a = ifelse(treatment_phase_2 == "Outpatient BUP + EMM", 1, -1),
         treatment_phase_1 = ifelse(treatment_phase_1 == "Outpatient BUP + EMM", 1, -1),
         is_male = ifelse(is_male == 1, 1, -1),
         reported_depression = ifelse(reported_depression == 1, 1, -1),
         moderate_to_severe_withdrawal_pre_induction = ifelse(
           moderate_to_severe_withdrawal_pre_induction == 1, 1, -1))

dat_x <- opioid_dat %>% 
  select(-c(who, treatment_phase_2, a, outcome)) %>%
  mutate(across(where(is.factor), ~as.numeric(.x)-1)) %>%
  as.matrix()

opioid_dat_formatted <- list(x=dat_x,
                           a=as.matrix(opioid_dat$a),
                           y=as.matrix(opioid_dat$outcome),
                           stage.x = rep(1, ncol(dat_x)))

estimated_regime <- estimate.regime.dl(opioid_dat_formatted$y, opioid_dat_formatted$a,
                                       opioid_dat_formatted$x, opioid_dat_formatted$stage.x,
                                       max.length=4L)

estimated_value <- estimated_regime$value

estimated_decision_list <- estimated_regime$dtr

print.regime(estimated_decision_list, colnames(opioid_dat_formatted$x))

assigned_values_under_optimal <- apply.regime.dl(estimated_regime$dtr[[1]],
                                                 opioid_dat_formatted$x)
saveRDS(assigned_values_under_optimal, "data/assigned_treatment_dl.RDS")

library(tidyverse)
tibble(optimal_assigned = assigned_values_under_optimal,
       true_assigned = as.numeric(opioid_dat_formatted$a)) %>%
  group_by(optimal_assigned, true_assigned) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(total = sum(n),
         prop=total/n)

tibble(optimal_assigned = assigned_values_under_optimal,
       true_assigned = as.numeric(opioid_dat_formatted$a)) %>%
  group_by(optimal_assigned) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(total = sum(n),
         prop=n/total)

tibble(optimal_assigned = assigned_values,
       true_assigned = as.numeric(opioid_dat_formatted$a)) %>%
  group_by(true_assigned) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(total = sum(n),
         prop=n/total)



get_value_dl <- function(data,ind) {
  
  data <- data[ind,]
  
 # glimpse(data)
  
  dat_x <- data %>% 
    select(-c(who, treatment_phase_2, a, outcome)) %>%
    mutate(across(where(is.factor), ~as.numeric(.x)-1)) %>%
    as.matrix()
  
  dat_formatted <- list(x=dat_x,
                        a=as.matrix(data$a),
                        y=as.matrix(data$outcome),
                        stage.x = rep(1, ncol(dat_x)))
  
  estimated_regime <- estimate.regime.dl(dat_formatted$y, dat_formatted$a,
                                         dat_formatted$x, dat_formatted$stage.x,
                                         max.length=10L,
                                         seed=NULL)
  
  estimated_value <- estimated_regime$value
  
  return(estimated_value)
  
}

library(moonboot)


m <- estimate.m(data=opioid_dat,
                statistic=get_value_dl,
                method="politis",
                R=100,
                replace=FALSE)
saveRDS(m, "data/optimal_decision_list_m.RDS")


value_dist <- map_dbl(1:500, ~ { 
  ind <- sample(1:nrow(opioid_dat), size = 140, replace = FALSE)
  get_value_dl(opioid_dat, ind)
})

dl_ci <- quantile(value_dist,c(.025,.975))
saveRDS(dl_ci, "data/optimal_decision_list_ci.RDS")




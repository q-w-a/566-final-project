
scripts_path <- here::here("supplementary_code_files/simulations")
source(paste0(scripts_path, "/regimes.R"))
source(paste0(scripts_path, "/sim_util.R"))

n <- 100        # sample size of a simulated dataset
scenario <- 5   # scenario number, 1 - 5

real_dat <- readRDS("/Users/quinnwhite/Desktop/spring_2025/566-final-project/data/final_data.RDS")

real_dat <- real_dat %>%
  mutate(a = ifelse(treatment == "Outpatient BUP + EMM", 1, -1))

dat_x <- real_dat %>% 
  select(-c(who, treatment, a, outcome)) %>%
  mutate(across(where(is.factor), ~as.numeric(.x)-1)) %>%
  as.matrix()

real_dat_formatted <- list(x=dat_x,
                           a=as.matrix(real_dat$a),
                           y=as.matrix(real_dat$outcome),
                           stage.x = rep(1, ncol(dat_x)))

estimated_regime <- estimate.regime.dl(real_dat_formatted$y, real_dat_formatted$a,
                   real_dat_formatted$x, real_dat_formatted$stage.x)

estimated_value <- estimated_regime$value

estimated_decision_list <- estimated_regime$dtr


est_value <- estimate.regime.dl(real_dat_formatted$y, real_dat_formatted$a,
                          real_dat_formatted$x, real_dat_formatted$stage.x)

est



a_optimal <- apply.regime.dl(est[[1]],
                             do.call("cbind", ))
  
  
for (k in 1L : ncol(real_dat_formatted$x)) {
    a[[k]] <- apply_regime(dtr[[k]], do.call("cbind",
                                             c(x[seq_len(k)], a[seq_len(k - 1L)], y[seq_len(k - 1L)])))
    
    
apply.regime.dl(est, real_dat_formatted$x)

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

print_regime(est, colnames(real_dat_formatted$x))



dl <- run.simulation(estimate.regime.dl, apply.regime.dl,
                     n = n, scenario = scenario)

dat <- generate.data(n, scenario)

variables <- dat$x %>% colnames()

print.regime(variables)

# what is stage.x? 

dl <- run.simulation(estimate.regime.dl, apply.regime.dl,
  n = n, scenario = scenario)
qlasso <- run.simulation(estimate.regime.qlasso, apply.regime.qlasso,
  n = n, scenario = scenario)
qrf <- run.simulation(estimate.regime.qrf, apply.regime.qrf,
  n = n, scenario = scenario)
bowl <- run.simulation(estimate.regime.bowl, apply.regime.bowl,
  n = n, scenario = scenario)
sowl <- run.simulation(estimate.regime.sowl, apply.regime.sowl,
  n = n, scenario = scenario)

# get mean and sd of the value achieved
mean(dl$metrics$outcome)
mean(qlasso$metrics$outcome)
mean(qrf$metrics$outcome)
mean(bowl$metrics$outcome)
mean(sowl$metrics$outcome)

sd(dl$metrics$outcome)
sd(qlasso$metrics$outcome)
sd(qrf$metrics$outcome)
sd(bowl$metrics$outcome)
sd(sowl$metrics$outcome)

# get average probability of correct treatment selection
mean(dl$metrics$correct.selection)
mean(qlasso$metrics$correct.selection)
mean(qrf$metrics$correct.selection)
mean(bowl$metrics$correct.selection)
mean(sowl$metrics$correct.selection)

# get average running time
mean(dl$metrics$duration)
mean(qlasso$metrics$duration)
mean(qrf$metrics$duration)
mean(bowl$metrics$duration)
mean(sowl$metrics$duration)



# load data 
opioid_dat <- readRDS("data/final_data.RDS") %>%
  mutate(a = ifelse(treatment_phase_2 == "Outpatient BUP + EMM", 1, 0),
         treatment_phase_1 = ifelse(treatment_phase_1 == "Outpatient BUP + EMM", 1, 0),
         is_male = ifelse(is_male == 1, 1, 0),
         reported_depression = ifelse(reported_depression == 1, 1, 0),
         moderate_to_severe_withdrawal_pre_induction = ifelse(
           moderate_to_severe_withdrawal_pre_induction == 1, 1, 0))

install.packages('DynTxRegime')
library(DynTxRegime)

main <- modelObj::buildModelObj(
  model = ~ treatment_phase_1 + prop_positive + 
    age + is_male + reported_depression + 
    moderate_to_severe_withdrawal_pre_induction,
  solver.method = 'glm',
  solver.args = list('family'="binomial"),
  predict.method = 'predict.glm',
  predict.args=list("type"="response"))
  
opt <- qLearn(moMain = main,
              moCont = main,
       response = opioid_dat$outcome,
       treatment = opioid_dat$treatment_phase_2,
       txName = "a",
       data = as.data.frame(opioid_dat),
       verbose=TRUE)


mod <- glm(outcome ~ a + a*(treatment_phase_1 + prop_positive + 
      age + is_male + reported_depression + 
      moderate_to_severe_withdrawal_pre_induction),
    data=opioid_dat,
    family=binomial) 

saveRDS(mod, "data/optimal_q_learning_model.RDS")

prediction_emm <- predict(
  mod, newdata = opioid_dat %>%
            mutate(a=1), type="response")

prediction_smm <- predict(
  mod, newdata = opioid_dat %>%
    mutate(a=0), type="response")

predictions <- tibble(prediction_emm, prediction_smm, 
       optimal = ifelse(prediction_emm > prediction_smm, "EMM", "SMM"),
                        est_val = pmax(prediction_emm, prediction_smm))

saveRDS(predictions$optimal, "data/assigned_treatment_ql.RDS")

# estimated value 
predictions %>%
  pull(est_val) %>%
  mean()


install.packages('moonboot')

library(moonboot)

get_value <- function(data, indices){
    
    data <- data[indices,]
    mod <- glm(outcome ~ a + a*(treatment_phase_1 + prop_positive + 
                                  age + is_male + reported_depression + 
                                  moderate_to_severe_withdrawal_pre_induction),
               data=data,
               family=binomial) 
    
    prediction_emm <- predict(
      mod, newdata = data %>%
        mutate(a=1), type="response")
    
    prediction_smm <- predict(
      mod, newdata = data %>%
        mutate(a=0), type="response")
    
    predictions <- tibble(prediction_emm, prediction_smm, 
                          optimal = ifelse(prediction_emm > prediction_smm, "EMM", "SMM"),
                          est_val = pmax(prediction_emm, prediction_smm))
    
    # estimated value 
    predictions %>%
      pull(est_val) %>%
      mean()
  
}


m <- estimate.m(data=opioid_dat,
           statistic=get_value,
           method="politis",
           R=200,
           replace=FALSE)

saveRDS(m, "data/optimal_q_learning_m.RDS")

value_dist <- map_dbl(1:500, ~ { 
  ind <- sample(1:nrow(opioid_dat), size = 155, replace = FALSE)
  get_value(opioid_dat, ind)
})

ql_ci <- quantile(value_dist,c(.025,.975))

saveRDS(ql_ci, "data/optimal_q_learning_ci.RDS")

# use m out of n bootstrap here to get confidence intervals 
# https://cran.r-project.org/web/packages/moonboot/moonboot.pdf

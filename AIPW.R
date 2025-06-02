
# Function 1
#------------------------------------------------------------------------------#
# components of the sandwich estimator for an ordinary least squares estimation
#   of a linear regression model
#
# ASSUMPTIONS
#   mo is a linear model with parameters to be estimated using OLS
#
# INPUTS
#    mo : a modeling object specifying a regression step
#         *** must be a linear model ***
#  data : a data.frame containing covariates and tx
#     y : a vector containing the outcome of interest
#
#
# RETURNS
# a list containing
#     MM : M^T M component from OLS
#  invdM : inverse of the matrix of derivatives of M wrt beta
#------------------------------------------------------------------------------#
swv_OLS <- function(mo, data, y) {
  
  n <- nrow(x = data)
  
  fit <- modelObj::fit(object = mo, data = data, response = y)
  
  # Q(X,A;betaHat)
  Qa <- drop(x = modelObj::predict(object = fit, newdata = data))
  
  # derivative of Q(X,A;betaHat)
  dQa <- stats::model.matrix(object = modelObj::model(object = mo), 
                             data = data)
  
  # number of parameters in model
  p <- ncol(x = dQa)
  
  # OLS component of M
  mOLSi <- {y - Qa}*dQa
  
  # OLS component of M MT
  mmOLS <- sapply(X = 1L:n, 
                  FUN = function(i){ mOLSi[i,] %o% mOLSi[i,] }, 
                  simplify = "array")
  
  # derivative OLS component
  dmOLS <- - sapply(X = 1L:n, 
                    FUN = function(i){ dQa[i,] %o% dQa[i,] }, 
                    simplify = "array")
  
  # sum over individuals
  if (p == 1L) {
    mmOLS <- mean(x = mmOLS)
    dmOLS <- mean(x = dmOLS)
  } else {
    mmOLS <- rowMeans(x = mmOLS, dim = 2L)
    dmOLS <- rowMeans(x = dmOLS, dim = 2L)
  }
  
  # invert derivative OLS component
  invD <- solve(a = dmOLS) 
  
  return( list("MM" = mmOLS, "invdM" = invD) )
  
}

# Function2
#------------------------------------------------------------------------------#
# components of the sandwich estimator for a maximum likelihood estimation of 
#   a logistic regression model
#
# ASSUMPTIONS
#   mo is a logistic model with parameters to be estimated using ML
#
# INPUTS
#    mo : a modeling object specifying a regression step
#         *** must be a logistic model ***
#  data : a data.frame containing covariates and tx
#     y : a vector containing the binary outcome of interest
#
#
# RETURNS
# a list containing
#     MM : M^T M component from ML
#  invdM : inverse of the matrix of derivatives of M wrt gamma
#------------------------------------------------------------------------------#
swv_ML <- function(mo, data, y) {
  
  # regression step
  fit <- modelObj::fit(object = mo, data = data, response = y)
  
  # yHat
  p1 <- modelObj::predict(object = fit, newdata = data) 
  if (is.matrix(x = p1)) p1 <- p1[,ncol(x = p1), drop = TRUE]
  
  # model.matrix for model
  piMM <- stats::model.matrix(object = modelObj::model(object = mo), 
                              data = data)
  
  n <- nrow(x = piMM)
  p <- ncol(x = piMM)
  
  # ML M-estimator component
  mMLi <- {y - p1} * piMM
  
  # ML component of M MT
  mmML <- sapply(X = 1L:n, 
                 FUN = function(i){ mMLi[i,] %o% mMLi[i,] }, 
                 simplify = "array")
  
  # derivative of ML component
  dFun <- function(i){ 
    - piMM[i,] %o% piMM[i,] * p1[i] * {1.0 - p1[i]}
  } 
  dmML <- sapply(X = 1L:n, FUN = dFun, simplify = "array")
  
  if( p == 1L ) {
    mmML <- mean(x = mmML)
    dmML <- mean(x = dmML)
  } else {
    mmML <- rowMeans(x = mmML, dim = 2L)
    dmML <- rowMeans(x = dmML, dim = 2L)
  }
  
  # invert derivative ML component
  invD <- solve(a = dmML) 
  
  return( list("MM" = mmML, "invdM" = invD) )
  
}

#Function 3
#------------------------------------------------------------------------------#
# AIPW estimator for value of a fixed binary tx regime
#
# ASSUMPTIONS
#   tx is binary coded as {0,1}
#
# INPUTS
#    moOR : modeling object specifying outcome regression
#    moPS : modeling object specifying propensity score regression
#    data : data.frame containing baseline covariates and tx
#           *** tx must be coded 0/1 ***
#       y : outcome of interest
#  txName : tx column in data (tx must be coded as 0/1)
#  regime : 0/1 vector containing recommended tx
#
# RETURNS
#  a list containing
#     fitOR : value object returned by outcome regression
#     fitPS : value object returned by propensity score regression
#        EY : sample average outcome for each tx received
#  valueHat : estimated value
#------------------------------------------------------------------------------#
value_AIPW <- function(moOR, moPS, data, y, txName, regime) {
  
  #### Propensity Score
  
  fitPS <- modelObj::fit(object = moPS, data = data, response = data[,txName])
  
  # estimated propensity score
  p1 <- modelObj::predict(object = fitPS, newdata = data)
  if (is.matrix(x = p1)) p1 <- p1[, ncol(x = p1), drop = TRUE]
  
  #### Outcome Regression
  
  fitOR <- modelObj::fit(object = moOR, data = data, response = y)
  
  # store tx variable
  A <- data[,txName]
  
  # estimated Q-function when all A=d
  data[,txName] <- regime
  Qd <- drop(x = modelObj::predict(object = fitOR, newdata = data))
  
  #### Value
  
  Cd <- regime == A
  pid <- p1*{regime == 1L} + {1.0-p1}*{regime == 0L}
  
  value <- Cd * y / pid - {Cd - pid} / pid * Qd
  
  EY <- array(data = 0.0, dim = 2L, dimnames = list(c("0","1")))
  EY[1L] <- mean(x = value*{A == 0L})
  EY[2L] <- mean(x = value*{A == 1L})
  
  value <- sum(EY)
  
  return( list(   "fitOR" = fitOR,
                  "fitPS" = fitPS,
                  "EY" = EY,
                  "valueHat" = value) )
  
}

#Function 4
#------------------------------------------------------------------------------#
# component of sandwich estimator for AIPW estimator of value of a fixed regime
#
# REQUIRES
#   value_AIPW()
#
# ASSUMPTIONS
#   tx is binary coded as {0,1}
#   moOR is a linear model
#   moPS is a logistic model
#
# INPUTS:
#    moOR : modeling object for outcome regression
#           *** must be a linear model ***
#    moPS : modeling object for propensity score regression
#           *** must be a logistic model ***
#    data : data.frame containing baseline covariates and tx
#           *** tx must be coded 0/1 ***
#       y : outcome of interest
#  txName : tx column in data (tx must be coded as 0/1)
#  regime : 0/1 vector containing recommended tx
#
# OUTPUTS:
#  value : list returned by value_AIPW()
#     MM : M M^T matrix
#   dMdB : matrix of derivatives of M wrt beta
#   dMdG : matrix of derivatives of M wrt gamma
#------------------------------------------------------------------------------#
value_AIPW_swv <- function(moOR, moPS, data, y, txName, regime) {
  
  # estimated value
  value <- value_AIPW(moOR = moOR, 
                      moPS = moPS, 
                      data = data,
                      y = y, 
                      regime = regime,
                      txName = txName)
  
  # pi(x;gammaHat)
  p1 <- modelObj::predict(object = value$fitPS, newdata = data)
  if( is.matrix(x = p1) ) p1 <- p1[,ncol(x = p1), drop=TRUE]
  
  # propensity to have received consistent tx
  pid <- p1*regime + {1.0-p1}*{1.0-regime}
  
  # model.matrix for propensity model
  piMM <- stats::model.matrix(object = modelObj::model(object = moPS), 
                              data = data)
  
  A <- data[,txName]
  
  # Q(H,A=d;betaHat)
  data[,txName] <- regime 
  Qd <- drop(modelObj::predict(object = value$fitOR, newdata = data))
  
  # dQ(H,A=d;betaHat)
  # derivative of a linear model is the model.matrix
  dQd <- stats::model.matrix(object = modelObj::model(object = moOR), 
                             data = data)
  
  Cd <- regime == A
  
  # value component of M-equation
  mValuei <- Cd * y / pid - {Cd - pid} / pid * Qd - value$valueHat
  mmValue <- mean(x = mValuei^2)
  
  # derivative of value component w.r.t. beta
  dMdB <- colMeans(x = -{Cd - pid} / pid*dQd)
  
  # derivative of value component w.r.t. gamma
  dMdG <- Cd*{y - Qd}/pid^2*{-1}^{regime}
  dMdG <- colMeans(x = dMdG*p1*{1.0-p1}*piMM) 
  
  return( list("value" = value, "MM" = mmValue, "dMdB" = dMdB, "dMdG" = dMdG) )
  
}

#Function 5
#------------------------------------------------------------------------------#
# value and standard error for the AIPW estimator of the value of a fixed
# regime using the sandwich estimator of variance
#
# REQUIRES
#   swv_ML(), swv_OLS(), and value_AIPW_swv()
#
# ASSUMPTIONS
#   tx is binary coded as {0,1}
#   moOR is a linear model
#   moPS is a logistic model
#
# INPUTS
#    moPS : modeling object specifying propensity score regression
#           *** must be a logistic model ***
#    moOR : modeling object specifying outcome regression
#           *** must be a linear model ***
#    data : data.frame containing covariates and tx
#           *** tx must be coded 0/1 ***
#       y : vector of outcome of interest
#  txName : treatment column in data (treatment must be coded as 0/1)
#  regime : 0/1 vector containing recommended tx
#
# RETURNS
#  a list containing
#     fitOR : value object returned by outcome regression
#     fitPS : value object returned by propensity score regression
#        EY : sample average outcome for each received tx grp
#  valueHat : estimated value
#  sigmaHat : estimated standard error
#------------------------------------------------------------------------------#
value_AIPW_se <- function(moPS, moOR, data, y, txName, regime) {
  
  #### ML components
  ML <- swv_ML(mo = moPS, data = data, y = data[,txName]) 
  
  #### OLS components
  OLS <- swv_OLS(mo = moOR, data = data, y = y) 
  
  #### estimator components
  AIPW <- value_AIPW_swv(moOR = moOR, 
                         moPS = moPS, 
                         data = data, 
                         y = y, 
                         regime = regime,
                         txName = txName)
  
  #### 1,1 Component of Sandwich Estimator
  
  ## ML contribution
  temp <- AIPW$dMdG %*% ML$invdM
  sig11ML <- temp %*% ML$MM %*% t(x = temp)
  
  ## OLS contribution
  temp <- AIPW$dMdB %*% OLS$invdM
  sig11OLS <- temp %*% OLS$MM %*% t(x = temp)
  
  sig11 <- drop(x = AIPW$MM + sig11ML + sig11OLS)
  
  return( c(AIPW$value, "sigmaHat" = sqrt(x = sig11 / nrow(x = data)))  )
  
}




################################################################################


df<- readRDS("data/final_data.RDS")
df <- as.data.frame(df)

library(modelObj)

#Create binary A1 — whether first stage is EMM (1 = EMM, 0 = SMM)

df$A1 <- ifelse(df$treatment_phase_1 == "Outpatient BUP + EMM", 1L,
                ifelse(df$treatment_phase_1 == "Outpatient BUP + SMM", 0L, NA_integer_))


# Define covariates vector
covariates <- c("prop_positive", "age", "is_male", 
                "reported_depression", "moderate_to_severe_withdrawal_pre_induction")

moPS <- buildModelObj(
  model         = as.formula(
    paste("A1 ~", paste(covariates, collapse = " + "))
  ),
  solver.method = glm,
  solver.args   = list(family = binomial(link = "logit")),
  predict.method = predict,
  predict.args   = list(type = "response")
)

moOR <- buildModelObj(
  model         = as.formula(
    paste("outcome ~ A1 +", paste(covariates, collapse = " + "))
  ),
  solver.method = "lm",
  solver.args   = NULL,
  predict.method = "predict",
  predict.args   = NULL
)


n <- nrow(df)
regime_SMM <- rep(0L, n)   # SMM  (0)
regime_EMM <- rep(1L, n)   # EMM  (1)


# For “all SMM”
res_SMM <- value_AIPW_se(
  moPS   = moPS,
  moOR   = moOR,
  data   = df,
  y      = df$outcome,
  txName = "A1",
  regime = regime_SMM
)

# For “all EMM” 
res_EMM <- value_AIPW_se(
  moPS   = moPS,
  moOR   = moOR,
  data   = df,
  y      = df$outcome,
  txName = "A1",
  regime = regime_EMM
)

cat("---- All SMM (regime = 0) ----\n",
    "Value     =", round(res_SMM[["valueHat"]],   4), "\n",
    "Std. Err. =", round(res_SMM[["sigmaHat"]], 4), "\n\n")

cat("---- All EMM (regime = 1) ----\n",
    "Value     =", round(res_EMM[["valueHat"]],   4), "\n",
    "Std. Err. =", round(res_EMM[["sigmaHat"]], 4), "\n\n")













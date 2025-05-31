library(glmnet)
library(randomForest)

dyn.load(paste0("supplementary_code_files/simulations/dtr", .Platform$dynlib.ext))
source(paste0(scripts_path, "/wsvm.R"))


##### DL #####

estimate.regime.dl <- function(y, a, x, stage.x, seed = 1988)
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
    max.length <- 10L
    
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




apply.regime.dl <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x) || ncol(current.x) < 2) {
    current.x <- cbind(current.x, 0)
  }

  action <- .Call("R_apply_rule", current.dtr, current.x) + 1L
  attr(current.dtr, "options")[action]
}




##### Q-learning with lasso #####

estimate.regime.qlasso <- function(y, a, x, stage.x)
{
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)
  dtr <- vector("list", n.stage)

  future.y <- double(n)

  for (i.stage in n.stage : 1L) {
    current.x <- cbind(x[, which(stage.x <= i.stage), drop = FALSE],
      a[, seq_len(i.stage - 1L), drop = FALSE],
      y[, seq_len(i.stage - 1L), drop = FALSE])
    current.a <- a[, i.stage]
    current.y <- y[, i.stage] + future.y
    
    options <- sort.int(unique(current.a))
    group <- factor(current.a, options)

    lasso.x <- lapply(levels(group), function(g) current.x * as.double(group == g))
    lasso.x <- do.call("cbind", lasso.x)
    
    intercept <- c(tapply(current.y, group, mean))
    lasso.y <- current.y
    for (g in levels(group)) {
      lasso.y[group == g] <- lasso.y[group == g] - intercept[g]
    }

    if (var(lasso.y) > 1e-8) {
      model <- cv.glmnet(lasso.x, lasso.y, intercept = FALSE, 
        nfolds = 5)
      beta <- as.double(coef(model, s = "lambda.min"))[-1L]
      beta <- matrix(beta, ncol(current.x), nlevels(group))
    } else {
      beta <- matrix(0, ncol(current.x), nlevels(group))
    }
    
    outcomes <- current.x %*% beta
    outcomes <- t(t(outcomes) + intercept)
    action <- max.col(outcomes, "first")
    
    dtr[[i.stage]] <- list(intercept = intercept, beta = beta)
    attr(dtr[[i.stage]], "options") <- options

    future.y <- outcomes[cbind(1L : n, action)]
  }
  
  dtr
}




apply.regime.qlasso <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x)) {
    current.x <- cbind(current.x)
  }

  d <- current.dtr
  outcomes <- current.x %*% d$beta
  outcomes <- t(t(outcomes) + d$intercept)
  action <- max.col(outcomes, "first")
  
  attr(d, "options")[action]
}




##### Q-learning with random forest #####

estimate.regime.qrf <- function(y, a, x, stage.x)
{
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)
  dtr <- vector("list", n.stage)

  future.y <- double(n)

  for (i.stage in n.stage : 1L) {
    current.x <- cbind(x[, which(stage.x <= i.stage), drop = FALSE],
      a[, seq_len(i.stage - 1L), drop = FALSE],
      y[, seq_len(i.stage - 1L), drop = FALSE])
    current.a <- a[, i.stage]
    current.y <- y[, i.stage] + future.y
    
    options <- sort.int(unique(current.a))
    group <- factor(current.a, options)

    colnames(current.x) <- NULL
    model <- lapply(levels(group), function(g)
      randomForest(current.x[group == g, , drop = FALSE],
        current.y[group == g]))

    outcomes <- sapply(model, function(m) predict(m, current.x))
    action <- max.col(outcomes, "first")
    
    dtr[[i.stage]] <- list(model = model)
    attr(dtr[[i.stage]], "options") <- options

    future.y <- outcomes[cbind(1L : n, action)]
  }
  
  dtr
}




apply.regime.qrf <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x)) {
    current.x <- cbind(current.x)
  }
  colnames(current.x) <- NULL
  
  d <- current.dtr
  outcomes <- sapply(d$model, function(m)
    predict(m, current.x))
  action <- max.col(outcomes, "first")
  
  attr(d, "options")[action]
}




##### BOWL #####

estimate.regime.bowl <- function(y, a, x, stage.x)
{
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)
  dtr <- vector("list", n.stage)
  options(warn = -1)

  future.w <- rep_len(1, n)

  cost.array <- 2 ^ (-5 : -15)
  num.folds <- 5L

  for (i.stage in n.stage : 1L) {
    current.x <- cbind(x[, which(stage.x <= i.stage), drop = FALSE],
      a[, seq_len(i.stage - 1L), drop = FALSE],
      y[, seq_len(i.stage - 1L), drop = FALSE])
    current.a <- a[, i.stage]
    current.y <- rowSums(y[, i.stage : n.stage, drop = FALSE])

    pr <- ifelse(current.a > 0, mean(current.a > 0), mean(current.a < 0))
    current.x.full <- current.x
    current.a.full <- current.a

    active <- (future.w > 0)
    current.x <- current.x[active, , drop = FALSE]
    current.a <- current.a[active]
    current.y <- current.y[active]
    
    options <- sort.int(unique(current.a))
    group <- factor(current.a, options)
    
    if (min(table(group)) >= 3) {
      colnames(current.x) <- NULL
      model <- lapply(levels(group), function(g)
        randomForest(current.x[group == g, , drop = FALSE],
                     current.y[group == g],
                     ntree = 50,
                     mtry = ceiling(ncol(current.x) * 0.25),
                     sampsize = ceiling(sum(group == g) * 0.8),
                     nodesize = 10,
                     importance = FALSE, localImp = FALSE, nPerm = 0))
      
      outcomes <- sapply(model, function(m) predict(m, current.x))
      baseline <- rowMeans(outcomes)
    } else {
      baseline <- current.y * 0
    }
    
    current.y <- (current.y - baseline) * future.w[active] / pr[active]
    temp <- ifelse(current.y >= 0, 1, -1)
    current.y <- current.y * temp
    current.a <- current.a * temp
    current.y <- current.y / mean(current.y)

    n <- length(current.y)
    fold <- sample.int(num.folds, n, TRUE)
    fold[current.y > 0] <- rep_len(seq_len(num.folds), sum(current.y > 0))

    val.array <- sapply(cost.array, function(cost) {
      cv <- 0
      for (ifold in seq_len(num.folds)) {
        i <- (fold != ifold)
        
        ## use fnscale = -1 since maximization problem
        fit <- wsvm(current.x[i, , drop = FALSE],
          current.a[i], current.y[i],
          scale = FALSE, kernel = "linear", type = "C",
          cost = cost)
        est <- coef(fit)
        pred <- as.double(est$intercept + 
          current.x[!i, , drop = FALSE] %*% est$beta)
        hinge <- pmax(1 - current.a[!i] * pred, 0)
        cv <- cv + sum(current.y[!i] * hinge)
      }
      cv
    })
    
    k <- which(val.array < min(val.array) * 1.000001)[1]
    cost <- cost.array[k]

    fit <- wsvm(current.x, current.a, current.y,
      scale = FALSE, kernel = "linear", type = "C",
      cost = cost)
    est <- coef(fit)
    est <- c(est$intercept, est$beta)
    
    dtr[[i.stage]] <- est
    
    rec <- ifelse(est[1] + current.x.full %*% est[-1] >= 0, 1, -1)
    future.w <- future.w * as.double(current.a.full == rec) / pr
  }
  
  options(warn = 0)
  dtr
}




apply.regime.bowl <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x)) {
    current.x <- cbind(current.x)
  }

  d <- current.dtr
  outcomes <- d[1] + current.x %*% d[-1]
  action <- ifelse(outcomes >= 0, 1, -1)
  
  action
}




##### SOWL #####

sowl2dObjFn <- function(b, x1, x2, a1, a2, y, lam1, lam2)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]
  beta1 = b[1:(p1+1)]
  beta2 = b[(p1+2):(p1+p2+2)]
  m1 = a1 * (beta1[1] + x1 %*% beta1[-1])
  m2 = a2 * (beta2[1] + x2 %*% beta2[-1])
  pen = lam1 * sum(beta1[-1]^2) + lam2 * sum(beta2[-1]^2)
  loss = sum(y * (pmin(m1, m2, 1)))
  loss - pen
}




sowl2dObjGr <- function(b, x1, x2, a1, a2, y, lam1, lam2)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]
  beta1 = b[1:(p1+1)]
  beta2 = b[(p1+2):(p1+p2+2)]
  m1 = a1 * (beta1[1] + x1 %*% beta1[-1])
  m2 = a2 * (beta2[1] + x2 %*% beta2[-1])
  pen.gr = c(0, 2 * lam1 * beta1[-1], 0, 2 * lam2 * beta2[-1])
  loss.gr = c(
    colSums(y * a1 * as.double(m1 <= m2 & m1 < 1) * cbind(1, x1)),
    colSums(y * a2 * as.double(m2 <= m1 & m2 < 1) * cbind(1, x2)))
  loss.gr - pen.gr
}




sowl2d <- function(x1, x2, a1, a2, y, lam.array)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]

  lam.array <- sort.int(lam.array, decreasing = TRUE)
  num.folds <- 5L
  fold <- sample.int(num.folds, length(y), TRUE)
  init <- matrix(rnorm(p1+p2+2,0,0.01), p1+p2+2, num.folds)
  
  val.array <- sapply(lam.array, function(lam) {
    cv <- 0
    for (ifold in seq_len(num.folds)) {
      i <- (fold != ifold)

      ## use fnscale = -1 since maximization problem
      train = optim(init[, ifold], sowl2dObjFn, sowl2dObjGr,
        control = list(fnscale = -1), method = "BFGS",
        x1 = x1[i, , drop = FALSE], x2 = x2[i, , drop = FALSE],
        a1 = a1[i], a2 = a2[i], y = y[i],
        lam1 = lam, lam2 = lam)$par

      init[, ifold] <- train
      cv <- cv + sowl2dObjFn(train,
        x1[!i, , drop = FALSE], x2[!i, , drop = FALSE],
        a1[!i], a2[!i], y[!i], 0, 0)
    }
    cv
  })
  
  k <- which(val.array > max(val.array) - 0.000001)[1]
  lam <- lam.array[k]

  bHat = optim(rnorm(p1+p2+2,0,0.01), sowl2dObjFn, sowl2dObjGr,
    control = list(fnscale = -1, maxit = 1000), method = "BFGS",
    x1 = x1, x2 = x2,
    a1 = a1, a2 = a2, y = y,
    lam1 = lam, lam2 = lam)$par

  list("bHat1" = bHat[1:(p1+1)],
       "bHat2" = bHat[(p1+2):(p1+p2+2)])
}




sowl3dObjFn <- function(b, x1, x2, x3, a1, a2, a3, y, lam1, lam2, lam3)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]
  p3 = dim(x3)[2]
  beta1 = b[1:(p1+1)]
  beta2 = b[(p1+2):(p1+p2+2)]
  beta3 = b[(p1+p2+3):(p1+p2+p3+3)]
  m1 = a1 * (beta1[1] + x1 %*% beta1[-1])
  m2 = a2 * (beta2[1] + x2 %*% beta2[-1])
  m3 = a3 * (beta3[1] + x3 %*% beta3[-1])
  pen = lam1 * sum(beta1[-1]^2) + lam2 * sum(beta2[-1]^2) + lam3 * sum(beta3[-1]^2)
  loss = sum(y * pmin(m1, m2, m3, 1))
  loss - pen
}




sowl3dObjGr <- function(b, x1, x2, x3, a1, a2, a3, y, lam1, lam2, lam3)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]
  p3 = dim(x3)[2]
  beta1 = b[1:(p1+1)]
  beta2 = b[(p1+2):(p1+p2+2)]
  beta3 = b[(p1+p2+3):(p1+p2+p3+3)]
  m1 = a1 * (beta1[1] + x1 %*% beta1[-1])
  m2 = a2 * (beta2[1] + x2 %*% beta2[-1])
  m3 = a3 * (beta3[1] + x3 %*% beta3[-1])
  pen.gr = c(0, 2 * lam1 * beta1[-1], 0, 2 * lam2 * beta2[-1], 0, 2 * lam3 * beta3[-1])
  loss.gr = c(
    colSums(y * a1 * as.double(m1 <= m2 & m1 <= m3 & m1 < 1) * cbind(1, x1)),
    colSums(y * a2 * as.double(m2 <= m1 & m2 <= m3 & m2 < 1) * cbind(1, x2)),
    colSums(y * a3 * as.double(m3 <= m1 & m3 <= m2 & m3 < 1) * cbind(1, x3)))
  loss.gr - pen.gr
}




sowl3d <- function(x1, x2, x3, a1, a2, a3, y, lam.array)
{
  p1 = dim(x1)[2]
  p2 = dim(x2)[2]
  p3 = dim(x3)[2]

  lam.array <- sort.int(lam.array, decreasing = TRUE)
  num.folds <- 5L
  fold <- sample.int(num.folds, length(y), TRUE)
  init <- matrix(rnorm(p1+p2+p3+3,0,0.01), p1+p2+p3+3, num.folds)
  
  val.array <- sapply(lam.array, function(lam) {
    cv <- 0
    for (ifold in seq_len(num.folds)) {
      i <- (fold != ifold)

      ## use fnscale = -1 since maximization problem
      train = optim(init[, ifold], sowl3dObjFn, sowl3dObjGr,
        control = list(fnscale = -1), method = "BFGS",
        x1 = x1[i, , drop = FALSE], x2 = x2[i, , drop = FALSE], x3 = x3[i, , drop = FALSE],
        a1 = a1[i], a2 = a2[i], a3 = a3[i], y = y[i],
        lam1 = lam, lam2 = lam, lam3 = lam)$par

      init[, ifold] <- train
      cv <- cv + sowl3dObjFn(train,
        x1[!i, , drop = FALSE], x2[!i, , drop = FALSE], x3[!i, , drop = FALSE],
        a1[!i], a2[!i], a3[!i], y[!i], 0, 0, 0)
    }
    cv
  })
  
  k <- which(val.array > max(val.array) - 0.000001)[1]
  lam <- lam.array[k]

  bHat = optim(rnorm(p1+p2+p3+3,0,0.01), sowl3dObjFn, sowl3dObjGr,
    control = list(fnscale = -1, maxit = 1000), method = "BFGS",
    x1 = x1, x2 = x2, x3 = x3,
    a1 = a1, a2 = a2, a3 = a3, y = y,
    lam1 = lam, lam2 = lam, lam3 = lam)$par

  list("bHat1" = bHat[1:(p1+1)],
       "bHat2" = bHat[(p1+2):(p1+p2+2)],
       "bHat3" = bHat[(p1+p2+3):(p1+p2+p3+3)])
}




estimate.regime.sowl <- function(y, a, x, stage.x)
{
  stopifnot(nrow(y) == nrow(a) && nrow(y) == nrow(x))
  n <- nrow(y)
  stopifnot(ncol(y) == ncol(a))
  n.stage <- ncol(y)

  future.y <- double(n)
  guess.action <- matrix(0, n, n.stage)
  guess.regret <- matrix(0, n, n.stage)
  for (i.stage in n.stage : 1L) {
    current.x <- cbind(x[, which(stage.x <= i.stage), drop = FALSE],
      a[, seq_len(i.stage - 1L), drop = FALSE],
      y[, seq_len(i.stage - 1L), drop = FALSE])
    current.a <- a[, i.stage]
    current.y <- y[, i.stage] + future.y
    
    options <- sort.int(unique(current.a))
    group <- factor(current.a, options)

    colnames(current.x) <- NULL
    model <- lapply(levels(group), function(g)
      randomForest(current.x[group == g, , drop = FALSE],
        current.y[group == g],
        ntree = 50,
        mtry = ceiling(ncol(current.x) * 0.25),
        sampsize = ceiling(sum(group == g) * 0.8),
        nodesize = 10,
        importance = FALSE, localImp = FALSE, nPerm = 0))

    outcomes <- sapply(model, function(m) predict(m, current.x))
    
    guess.action[, i.stage] <- max.col(outcomes, "first")
    future.y <- outcomes[cbind(1L : n, guess.action[, i.stage])]
    bad.action <- max.col(-outcomes, "first")
    bad.y <- outcomes[cbind(1L : n, bad.action)]
    guess.action[, i.stage] <- options[guess.action[, i.stage]]
    guess.regret[, i.stage] <- future.y - bad.y
  }
  
  y <- y + (guess.action != a) * guess.regret
  a <- guess.action

  if (n.stage == 2L) {
    
    x1 <- x[, stage.x <= 1L]
    x2 <- cbind(x[, stage.x <= 2L], a[, 1L], y[, 1L])
    a1 <- a[, 1L]
    a2 <- a[, 2L]
    y <- rowSums(y)
    y <- y - min(y) + 0.0001
    y <- y / mean(y)
    dtr <- sowl2d(x1, x2, a1, a2, y, n * 4 ^ (4 : -5))
    
  } else if (n.stage == 3L) {
    
    x1 <- x[, stage.x <= 1L]
    x2 <- cbind(x[, stage.x <= 2L], a[, 1L], y[, 1L])
    x3 <- cbind(x[, stage.x <= 3L], a[, 1L : 2L], y[, 1L : 2L])
    a1 <- a[, 1L]
    a2 <- a[, 2L]
    a3 <- a[, 3L]
    y <- rowSums(y)
    y <- y - min(y) + 0.0001
    y <- y / mean(y)
    dtr <- sowl3d(x1, x2, x3, a1, a2, a3, y, n * 4 ^ (5 : -4))
    
  } else {
    
    dtr <- NULL

  }
  
  dtr
}




apply.regime.sowl <- function(current.dtr, current.x)
{
  if (!is.matrix(current.x)) {
    current.x <- cbind(current.x)
  }
  
  d <- current.dtr
  outcomes <- d[1] + current.x %*% d[-1]
  action <- ifelse(outcomes >= 0, 1, -1)
  
  action
}




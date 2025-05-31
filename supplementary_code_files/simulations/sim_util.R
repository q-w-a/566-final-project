gh_quad <- function(f)
{
  # Gauss-Hermite quadrature
  # compute \inf_{-\infty}^{\infty} f(u) * dnorm(u) du

  gh_x <- c(
    6.889122439895, 5.744460078659, 4.778531589630, 3.900065717198, 
    3.073797175328, 2.281019440253, 1.509883307797, 0.751842600704, 
    -0.751842600704, -1.509883307797, -2.281019440253, -3.073797175328, 
    -3.900065717198, -4.778531589630, -5.744460078659, -6.889122439895)
  gh_w <- c(
    0.000000000026, 0.000000028080, 0.000004012679, 0.000168491432, 
    0.002858946062, 0.023086657026, 0.097406371163, 0.226706308469, 
    0.226706308469, 0.097406371163, 0.023086657026, 0.002858946062, 
    0.000168491432, 0.000004012679, 0.000000028080, 0.000000000026)
  gh_w0 <- 0.299538370127

  val <- gh_w0 * f(0)
  for (i in 1:16)
    val <- val + gh_w[i] * f(gh_x[i])
  val
}


generate.data <- function(n, scenario, dtr = NULL, apply_regime = NULL)
{
  # generate data from a SMART trial (if dtr is NULL)
  # or data following the given dtr (if dtr is a list)
  # or data following the optimal regime (if dtr is "optimal")

  if (scenario == 1L) {

    x1 <- matrix(rnorm(n * 50, 0, 1), n, 50)
    if (is.list(dtr)) {
      a1 <- apply_regime(dtr[[1]], x1)
    } else if (is.character(dtr) && dtr == "optimal") {
      tmp1pos <- 0.5 * x1[, 3] * 1
      tmp1neg <- 0.5 * x1[, 3] * -1
      tmp2 <- (x1[, 1] ^ 2 + x1[, 2] ^ 2 - 0.2) * 
        (0.5 - x1[, 1] ^ 2 - x1[, 2] ^ 2)
      val1pos <- gh_quad(function(u) 
        ((tmp1pos + u) + abs(tmp2 + tmp1pos + u)))
      val1neg <- gh_quad(function(u) 
        ((tmp1neg + u) + abs(tmp2 + tmp1neg + u)))
      a1 <- ifelse(val1pos >= val1neg, 1, -1)
    } else {
      a1 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y1 <- rnorm(n, 0.5 * x1[, 3] * a1, 1)
  
    x2 <- matrix(0, n, 0)
    if (is.list(dtr)) {
      a2 <- apply_regime(dtr[[2]], cbind(x1, x2, a1, y1))
    } else if (is.character(dtr) && dtr == "optimal") {
      a2 <- 2 * rbinom(n, 1, 0.5) - 1
      a2 <- ifelse((x1[, 1] ^ 2 + x1[, 2] ^ 2 - 0.2) * 
        (0.5 - x1[, 1] ^ 2 - x1[, 2] ^ 2) + y1 >= 0, 1, -1)
    } else {
      a2 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y2 <- rnorm(n, ((x1[, 1] ^ 2 + x1[, 2] ^ 2 - 0.2) * 
      (0.5 - x1[, 1] ^ 2 - x1[, 2] ^ 2) + y1) * a2, 1)

    x <- list(x1, x2)
    a <- list(a1, a2)
    y <- list(y1, y2)

  } else if (scenario == 2L) {

    x1 <- matrix(rnorm(n * 50, 0, 1), n, 50)
    if (is.list(dtr)) {
      a1 <- apply_regime(dtr[[1]], x1)
    } else if (is.character(dtr) && dtr == "optimal") {
      tmp1 <- 1 + 1.5 * x1[, 3]
      tmp21 <- 1.25 * x1[, 1]
      tmp22 <- -1.75 * x1[, 2]
      pr1pos <- pnorm(0, tmp21 * 1, 1, lower.tail = FALSE)
      pr2pos <- pnorm(0, tmp22 * 1, 1, lower.tail = FALSE)
      pr1neg <- pnorm(0, tmp21 * -1, 1, lower.tail = FALSE)
      pr2neg <- pnorm(0, tmp22 * -1, 1, lower.tail = FALSE)
      tmpf <- function(u, a1) {
        if (a1 > 0) {
          pr1 <- pr1pos
          pr2 <- pr2pos
        } else {
          pr1 <- pr1neg
          pr2 <- pr2neg
        }
        z1 <- tmp1 * a1 + u
        z2 <- 0.5 + z1 + 0.5 * a1
        z1 + abs(z2 + 0.5 * 1 - 0.5 * 1) * pr1 * pr2 +
          abs(z2 + 0.5 * 1 - 0.5 * 0) * pr1 * (1 - pr2) +
          abs(z2 + 0.5 * 0 - 0.5 * 1) * (1 - pr1) * pr2 +
          abs(z2 + 0.5 * 0 - 0.5 * 0) * (1 - pr1) * (1 - pr2)
      }
      val1pos <- gh_quad(function(u) tmpf(u, 1))
      val1neg <- gh_quad(function(u) tmpf(u, -1))
      a1 <- ifelse(val1pos >= val1neg, 1, -1)
    } else {
      a1 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y1 <- rnorm(n, (1 + 1.5 * x1[, 3]) * a1, 1)
    
    x2 <- cbind(as.double(rnorm(n, 1.25 * x1[, 1] * a1, 1) > 0), 
      as.double(rnorm(n, -1.75 * x1[, 2] * a1, 1) > 0))
    if (is.list(dtr)) {
      a2 <- apply_regime(dtr[[2]], cbind(x1, x2, a1, y1))
    } else if (is.character(dtr) && dtr == "optimal") {
      a2 <- ifelse(0.5 + y1 + 0.5 * a1 +
        0.5 * x2[, 1] - 0.5 * x2[, 2] >= 0, 1, -1)
    } else {
      a2 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y2 <- rnorm(n, (0.5 + y1 + 0.5 * a1 +
      0.5 * x2[, 1] - 0.5 * x2[, 2]) * a2, 1)
  
    x <- list(x1, x2)
    a <- list(a1, a2)
    y <- list(y1, y2)

  } else if (scenario == 3L || scenario == 4L) {

    x1 <- matrix(rnorm(n * 50, 0, 1), n, 50)
    x1 <- 45 + 15 * x1
    if (scenario == 3L) x1 <- x1[, 1 : 3]
    if (is.list(dtr)) {
      a1 <- apply_regime(dtr[[1]], x1)
    } else if (is.character(dtr) && dtr == "optimal") {
      a1 <- ifelse(x1[, 1] > 30, 1, -1)
    } else {
      a1 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y1 <- double(n)
    
    x2 <- cbind(rnorm(n, 1.5 * x1[, 1], 10))
    if (is.list(dtr)) {
      a2 <- apply_regime(dtr[[2]], cbind(x1, x2, a1, y1))
    } else if (is.character(dtr) && dtr == "optimal") {
      a2 <- ifelse(x2[, 1] > 40, 1, -1)
    } else {
      a2 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y2 <- double(n)
    
    x3 <- cbind(rnorm(n, 0.5 * x2[, 1], 10))
    if (is.list(dtr)) {
      a3 <- apply_regime(dtr[[3]], cbind(x1, x2, x3, a1, a2, y1, y2))
    } else if (is.character(dtr) && dtr == "optimal") {
      a3 <- ifelse(x3[, 1] > 40, 1, -1)
    } else {
      a3 <- 2 * rbinom(n, 1, 0.5) - 1
    }
    y3 <- rnorm(n, 20 - 
      abs(0.6 * x1[, 1] - 40) *
        (as.double(a1 > 0) - as.double(x1[, 1] > 30)) ^ 2 -
      abs(0.8 * x2[, 1] - 60) *
        (as.double(a2 > 0) - as.double(x2[, 1] > 40)) ^ 2 - 
      abs(1.4 * x3[, 1] - 40) *
        (as.double(a3 > 0) - as.double(x3[, 1] > 40)) ^ 2, 1)

    x <- list(x1, x2, x3)
    a <- list(a1, a2, a3)
    y <- list(y1, y2, y3)

  } else if (scenario == 5L) {

    x <- vector("list", 10L)
    a <- vector("list", 10L)
    y <- vector("list", 10L)

    for (k in 1L : 10L) {
      u <- rnorm(n, 0, 0.1)
      if (k == 1L) {
        x[[k]] <- 0.5 + u
      } else {
        x[[k]] <- 0.5 + 0.2 * x[[k - 1L]] - 0.08 * abs(a[[k - 1L]]) + u
      }

      if (is.list(dtr)) {
        a[[k]] <- apply_regime(dtr[[k]], do.call("cbind",
          c(x[seq_len(k)], a[seq_len(k - 1L)], y[seq_len(k - 1L)])))
      } else if (is.character(dtr) && dtr == "optimal") {
        a[[k]] <- ifelse(x[[k]] > 5 / 9,
          c(-1, -2, -3)[max.col(
            -cbind(1 - 2 * x[[k]], 2 - 2 * x[[k]], 3 - 2 * x[[k]]) ^ 2,
            ties.method = "first")],
          c(0, 1, 2, 3)[max.col(
            -cbind(0 - 5.5 * x[[k]], 1 - 5.5 * x[[k]], 2 - 5.5 * x[[k]],
            3 - 5.5 * x[[k]]) ^ 2, ties.method = "first")])
      } else {
        a[[k]] <- sample.int(7L, n, TRUE,
          c(rep_len(1 / 6, 3), rep(1 / 8, 4))) - 4L
      }

      temp1 <- as.double(a[[k]] < 0L) - as.double(x[[k]] > 5 / 9)
      temp2 <- as.double(a[[k]] < 0L) * (-a[[k]] - 2 * x[[k]])
      temp3 <- as.double(a[[k]] >= 0L) * (a[[k]] - 5.5 * x[[k]])
      y[[k]] <- (-5 * u - 6 * temp1 ^ 2 -
        1.5 * temp2 ^ 2 - 1.5 * temp3 ^ 2 + rnorm(n, 0, 0.8))
      if (k == 1L) {
        y[[k]] <- y[[k]] + 30
      }
    }

  } else {
    stop("Invalid 'scenario'.")
  }
  
  stage.x <- rep(seq_along(x), times = sapply(x, NCOL))
  x <- do.call("cbind", x)
  a <- do.call("cbind", a)
  y <- do.call("cbind", y)

  colnames(x) <- paste0("x", seq_len(ncol(x)))
  colnames(a) <- paste0("a", seq_len(ncol(a)))
  colnames(y) <- paste0("y", seq_len(ncol(y)))
  
  list(y = y, a = a, x = x, stage.x = stage.x)
}




run.simulation <- function(estimate_regime, apply_regime,
  n = 100L, scenario = 1L, r = 1000L)
{
  metrics <- data.frame(outcome = double(r), optimal.outcome = double(r),
    correct.selection = double(r), duration = double(r))
  rule <- vector("list", r)

  for (i in seq_len(r)) {
    set.seed(i)
    data <- generate.data(n, scenario)
    y <- data$y
    a <- data$a
    x <- data$x
    stage.x <- data$stage.x

    start.time <- proc.time()[3]
    dtr <- estimate_regime(y, a, x, stage.x)
    end.time <- proc.time()[3]
    
    set.seed(10000 + i)
    test.data <- generate.data(1e5, scenario, dtr, apply_regime)
    set.seed(10000 + i)
    optimal.data <- generate.data(1e5, scenario, "optimal")

    # avoid out of memory
    if (object.size(dtr) <= 51200) {
      rule[[i]] <- dtr
    }

    metrics$outcome[i] <- mean(rowSums(test.data$y))
    metrics$optimal.outcome[i] <- mean(rowSums(optimal.data$y))
    metrics$correct.selection[i] <- 
      mean(rowSums(test.data$a == optimal.data$a) == ncol(test.data$a))
    metrics$duration[i] <- end.time - start.time

    print(i)
  }
  
  list(metrics = metrics, rule = rule)
}




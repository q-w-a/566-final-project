#ifndef DTR_H_INCLUDED
#define DTR_H_INCLUDED




/*
    Overview
    gaussian_kernel.c: RKHS regression with Gaussian kernel
    minimization.c: given covariates and action, find cutoff values
    clause.c: find covariates to use and action to take
    rule.c: find if-then list, which consists of several clauses
*/




/** gaussian_kernel.c **/

typedef struct {
    int n;                      /* number of samples in this group */
    double percent;             /* percent of samples in this group */
    double intercept;           /* mean of y */
    double *restrict x;         /* n by p */
    double *restrict npdx;      /* n * (n + 1) / 2 by p */
    double *restrict k;         /* n by n */
    double *restrict kpinv;     /* n by n */
    double *restrict alpha;     /* n */
    double *restrict y;         /* n */
    double *restrict work;      /* n by (4 + 2 * n) */
} training_data;

typedef struct {
    int p;                      /* number of predictors */
    training_data *data;        /* num_groups */
    int num_groups;
    double *restrict scaling;   /* p */
    double *restrict gamma;     /* p */
    double lambda;
    double *restrict work;      /* p + 1 */
} training_input;

void kernel_train(const int n, const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling,
    double *restrict node, double *restrict alpha,
    int *restrict offset, double *restrict gamma,
    double *restrict intercept);

void kernel_predict(const int p, const int num_groups, const int new_n,
    const double *restrict node, const double *restrict alpha,
    const int *restrict offset, const double *restrict gamma,
    const double *restrict intercept, const double *restrict new_x,
    double *restrict new_yhat);




/** outcomes.c **/

void get_regrets_from_outcomes(const double *restrict outcomes,
    const unsigned int n, const unsigned int m,
    double *restrict regrets);




/** sort.c **/

void sort_matrix(const double *restrict z,
    unsigned int *restrict iz,
    unsigned int *restrict rz,
    double *restrict sz,
    const unsigned int n, const unsigned int p);

void reverse_sort(const unsigned int *restrict iz,
    const unsigned int *restrict rz,
    const double *restrict sz,
    unsigned int *restrict rev_iz,
    unsigned int *restrict rev_rz,
    double *restrict rev_sz,
    const unsigned int n, const unsigned int p);




/** minimization.c **/

void minimize_loss(const unsigned int *restrict ix,
    const unsigned int *restrict rx,
    const double *restrict sx,
    const unsigned int *restrict iy,
    const unsigned int *restrict ry,
    const double *restrict sy,
    const double *restrict loss,
    const unsigned int *restrict next,
    const unsigned int n, const double zeta, const double eta,
    double *restrict opt_cx, double *restrict opt_cy,
    double *restrict opt_loss);




/** clause.c **/

#define CONDITION_TYPE_LL        1
#define CONDITION_TYPE_LR        2
#define CONDITION_TYPE_RL        3
#define CONDITION_TYPE_RR        4

typedef struct {
    unsigned int a;
    unsigned char type;
    unsigned int j1, j2;
    double c1, c2;
} statement;

void find_clause(const unsigned int *restrict asc_iz,
    const unsigned int *restrict asc_rz,
    const double *restrict asc_sz,
    const unsigned int *restrict desc_iz,
    const unsigned int *restrict desc_rz,
    const double *restrict desc_sz,
    const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    statement *restrict clause);

void find_last_clause(const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int m,
    statement *restrict clause);

void apply_clause(const double *restrict z,
    unsigned int *restrict next, const unsigned int n,
    const statement *restrict clause, int *restrict action);




/** rule.c **/

void find_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    const unsigned int max_length,
    statement *restrict rule, unsigned int *restrict rule_length,
    int *restrict action);

void apply_rule(const double *restrict z,
    const unsigned int n,
    const statement *rule, const unsigned int rule_length,
    int *restrict action);

void cv_tune_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const int *restrict fold, const unsigned int num_folds,
    double *restrict cv_regret);

void batch_evaluate_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const double *restrict test_z,
    const double *restrict test_regrets,
    const unsigned int test_n,
    double *restrict test_mean_regret);




#endif /* DTR_H_INCLUDED */



#ifndef PROJECTED_BFGS_H_INCLUDED
#define PROJECTED_BFGS_H_INCLUDED




#define MX_BFGS_ACCEPT      0.0001
#define MX_BFGS_SHRINKAGE   0.2
#define MX_BFGS_STRETCH     2
#define MX_BFGS_LONGEST     100
#define MX_BFGS_TINY        1e-20
#define MX_BFGS_NOT_UPDATE  0.25
#define MX_BFGS_RESTART     1.2




/* x, p, extra */
typedef double (*objective_function)(const double *, int, void *);
/* x, p, extra, gradient */
typedef void (*objective_gradient)(const double *, int, void *, double *);




void projected_bfgs(double *restrict x, const int p, void *extra,
    objective_function func, objective_gradient grad,
    const int max_iter, const double tol,
    const double *restrict lower, const double *restrict upper,
    const double step_limit, double *value);




#endif /* PROJECTED_BFGS_H_INCLUDED */



#include <stdlib.h>
#include <math.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>




void projected_bfgs(double *restrict x, const int p, void *extra,
    objective_function func, objective_gradient grad,
    const int max_iter, const double tol,
    const double *restrict lower, const double *restrict upper,
    const double step_limit, double *value)
{
    /*
    BFGS algorithm for minimization with box constraints
    reference: Kim, D., Sra, S. & Dhillon, I. S. (2010).
    Tackling box-constrained optimization via a new
    projected quasi-Newton approach.
    SIAM Journal on Scientific Computing, 32, 3548?3563

    x: vector of length p, initial value
    extra: addition parameters needed by func
    func: objective function
    grad: gradient of objective function
    max_iter: number of maximum iterations in BFGS
    tol: relative tolerance for convergence in minimum value
    lower: vector of length p, lower bound constraints
    upper: vector of length p, upper bound constraints
    step_limit: largest step size allowed
    value: scalar, minimum value
    */

    const int q1 = p * sizeof(double);
    const int q2 = (p * (p + 1) / 2) * sizeof(double);
    int *restrict bind = (int *)malloc(p * sizeof(int));
    if (bind == 0) return;

    #define project_to_box                       \
        for (int k = 0; k < p; ++k) {            \
            if (x[k] <= lower[k]) {              \
                x[k] = lower[k]; bind[k] = 1;    \
            } else if (x[k] >= upper[k]) {       \
                x[k] = upper[k]; bind[k] = 2;    \
            } else {                             \
                bind[k] = 0;                     \
            }                                    \
        }
    project_to_box;

    double f = func(x, p, extra);
    *value = f;

    const int lwork = p * 4 + p * (p + 1) / 2;
    double *work = (double *)malloc(lwork * sizeof(double));
    if (work == 0) return;

    double *restrict g = work;          /* p-vector */
    double *restrict s = work + p;      /* p-vector */
    double *restrict y = work + 2 * p;  /* p-vector */
    double *restrict u = work + 3 * p;  /* p-vector */
    double *restrict H = work + 4 * p;  /* p by p symmetric matrix */
                                        /* packed form, upper part */
    grad(x, p, extra, g);

    const int inc_one = 1;
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const char upper_part = 'U';

    const int limit_not_updateH = (int)(p * MX_BFGS_NOT_UPDATE) + 1;
    const int limit_restartH = (int)(p * MX_BFGS_RESTART) + 1;
    int i = 1, j, not_updateH = 0, restartH = 0, fixed;
    int resetH = 1; /* whether to set H as a scaled identity matrix */
    int accept = 0; /* whether to accept a point in line search */
    double dotprod, f1, rho, temp;
    double gamma = 1.0;       /* scaling factor for H */
    double a, min_a;          /* step length in line search */
    double step_size;         /* l1 norm of s */

    while ((i <= max_iter) && (resetH != -1)) {

        /* periodical restart */
        if (restartH == limit_restartH) resetH = 1;

        /* search direction: s = - H_{free} g */
        memcpy(u, g, q1);
        for (int k = 0; k < p; ++k) {
            if (((bind[k] == 1) && (u[k] > 0.0))
                || ((bind[k] == 2) && (u[k] < 0.0))) {
                bind[k] = 3; u[k] = 0.0;
            }
        }

        if (resetH == 0) {
            ++restartH;
            F77_CALL(dspmv)(&upper_part, &p, &neg_one, H,
                u, &inc_one, &zero, s, &inc_one, 1);
        } else {
            restartH = 0;
            /* s = -g */
            memcpy(s, u, q1);
            F77_CALL(dscal)(&p, &neg_one, s, &inc_one);
        }

        step_size = 0.0;
        for (int k = 0; k < p; ++k)
            step_size += fabs(s[k]);

        fixed = 0;
        for (int k = 0; k < p; ++k) {
            if ((bind[k] == 3)
                || ((bind[k] == 1) && (s[k] < 0.0))
                || ((bind[k] == 2) && (s[k] > 0.0))) {
                s[k] = 0.0; ++fixed;
            }
        }

        /* dotprod = s^T g */
        if (fixed < p)
            dotprod = F77_CALL(ddot)(&p, s, &inc_one, g, &inc_one);
        else
            dotprod = 1.0; /* in this case s is a zero vector */

        /* checks whether s is a downhill direction */
        if (dotprod < 0.0) {
            /* performs a line search along s */
            a = step_limit / step_size;
            if (a > 1.0) a = 1.0;

            min_a = (fabs(f) * MX_BFGS_ACCEPT * tol + MX_BFGS_TINY)
                / (-dotprod);
            if (min_a > 1.0) min_a = 1.0;

            accept = 0; memcpy(u, x, q1);
            while ((!accept) && (a >= min_a)) {
                /* x = u + a s */
                F77_CALL(daxpy)(&p, &a, s, &inc_one, x, &inc_one);
                project_to_box;
                f1 = func(x, p, extra);
                accept = (f1 <= f + MX_BFGS_ACCEPT * dotprod * a);
                if (!accept) {
                    a *= MX_BFGS_SHRINKAGE; memcpy(x, u, q1);
                }
            }

            /* checks convergence */
            if (accept) {
                temp = f - tol * (fabs(f) + MX_BFGS_TINY);
                f = f1;
                if (f >= temp) accept = 0;
            }

            /* checks sufficient descent */
            if (accept) {
                ++i;
                /* s = x0 - x1 */
                F77_CALL(daxpy)(&p, &neg_one, x, &inc_one, u, &inc_one);
                memcpy(s, u, q1);
                /* y = g0 - g1 */
                memcpy(y, g, q1);
                grad(x, p, extra, g);
                F77_CALL(daxpy)(&p, &neg_one, g, &inc_one, y, &inc_one);
                /* rho = s^T y */
                rho = F77_CALL(ddot)(&p, s, &inc_one, y, &inc_one);

                /* to keep positive definiteness, rho must > 0 */
                if (rho > 0.0) {
                    /* u = H y */
                    if (resetH == 0) {
                        F77_CALL(dspmv)(&upper_part, &p, &one, H,
                            y, &inc_one, &zero, u, &inc_one, 1);
                    } else {
                        /* gamma = s^T y / y^T y */
                        temp = F77_CALL(dnrm2)(&p, y, &inc_one);
                        gamma = rho / (temp * temp);
                        /* u = gamma * y */
                        memcpy(u, y, q1);
                        F77_CALL(dscal)(&p, &gamma, u, &inc_one);

                        /* sets H to scaled identity matrix: H = gamma I */
                        memset(H, 0, q2);
                        for (j = 0; j < p; ++j)
                            H[j * (j + 3) / 2] = gamma;
                    }

                    /* performs the BFGS update of H */
                    temp = F77_CALL(ddot)(&p, y, &inc_one, u, &inc_one);
                    temp = (1.0 + temp / rho) / rho;
                    F77_CALL(dspr)(&upper_part, &p, &temp, s, &inc_one, H,1);
                    temp = -1.0 / rho;
                    F77_CALL(dspr2)(&upper_part, &p, &temp, s, &inc_one,
                        u, &inc_one, H,1);

                    resetH = 0;
                } else { /* i.e. rho <= 0 */
                    ++not_updateH;
                    if (not_updateH >= limit_not_updateH) {
                        resetH = 1; not_updateH = 0;
                    }
                }
            } else { /* i.e. !accept */
                if (resetH == 0) resetH = 1; else resetH = -1;
            }
        } else { /* i.e. dotprod >= 0 */
            if (resetH == 0) resetH = 1; else resetH = -1;
        }

    } /* while ((i <= max_iter) && (resetH != -1)) */

    *value = f;
    free(work);
    free(bind);
}



#include <stdlib.h>
#include <math.h>
#include <R_ext/Lapack.h>




/********************************/
/* training in a single dataset */
/********************************/




/* index of the i-th diagonal element in packed format */
#define DIAG_ELEM(i) ((i * (i + 3)) / 2)




void transform_x(const int n, const int p,
    const double *restrict x,
    double *restrict npdx)
{
    /*
    for each j, compute (x_{ij} - x_{i'j})^2 for all pairs i <= i'

    x: matrix of size n by p, predictors, each row is an x_i
    npdx (negative pairwise distance in x): matrix of size n (n + 1) / 2 by p,
        in upper triangle packed format, thus npdx[i1, i2] is stored in
        the (i1 + i2 (i2 + 1) / 2)-th element for i1, i2 = 0, ..., n - 1
    */

    const int nrow_npdx = (n * (n + 1)) / 2;
    int offset_x, offset_npdx;
    double temp_x2, temp_d;

    for (int j = 0; j < p; ++j) {
        offset_x = n * j;
        for (int i2 = 0; i2 < n; ++i2) {
            offset_npdx = nrow_npdx * j + (i2 * (i2 + 1)) / 2;
            temp_x2 = x[i2 + offset_x];

            for (int i1 = 0; i1 < i2; ++i1) {
                temp_d = x[i1 + offset_x] - temp_x2;
                npdx[i1 + offset_npdx] = -temp_d * temp_d;
            }
            npdx[i2 + offset_npdx] = 0.0;
        }
    }
}




void center_y(const int n,
    double *restrict y, double *restrict intercept)
{
    /*
    center y to have mean zero

    y: vector of length n, responses
    intercept: scalar, sample mean of y
    */

    double mean = 0.0;
    for (int i = 0; i < n; ++i)
        mean += y[i];
    mean /= n;

    for (int i = 0; i < n; ++i)
        y[i] -= mean;
    *intercept = mean;
}




void compute_kernel(const int n, const int p,
    const double *restrict npdx,
    const double *restrict gamma,
    const double lambda,
    double *restrict k,
    double *restrict kpinv)
{
    /*
    compute the design matrix defined by gaussian kernel as well as
    its inverse

    npdx: matrix of size n (n + 1) / 2 by p
    gamma: vector of length p, scaling factors
    lambda: scalar
    k: matrix of size n by n, design matrix induced by gaussian kernel,
        in upper triangle packed format
    kpinv: inverse of (k + lambda * I_n)
    */

    const char normal = 'N';
    const int inc_one = 1;
    const double zero = 0.0, one = 1.0;
    const int nrow_npdx = (n * (n + 1)) / 2;

    /* DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY) */
    F77_NAME(dgemv)(&normal, &nrow_npdx, &p, &one, npdx, &nrow_npdx,
        gamma, &inc_one, &zero, k, &inc_one,1);

    for (int i = 0; i < nrow_npdx; ++i)
        k[i] = exp(k[i]);

    /* adding lambda to the diagonal */
    memcpy(kpinv, k, nrow_npdx * sizeof(double));
    for (int j = 0; j < n; ++j)
        kpinv[DIAG_ELEM(j)] += lambda;

    /* DPPTRF(UPLO,N,AP,INFO) */
    const char upper_part = 'U';
    int info = 0;
    F77_NAME(dpptrf)(&upper_part, &n, kpinv, &info, 1);

    /* DPPTRI(UPLO,N,AP,INFO) */
    F77_NAME(dpptri)(&upper_part, &n, kpinv, &info, 1);
}




void estimate_alpha(const int n,
    const double *restrict kpinv,
    const double *restrict y,
    double *restrict alpha)
{
    /*
    compute the regression coefficients (called alpha)
    let K be the kernel matrix indexed by gamma and lambda be a parameter
    alpha = argmin_{a} (y - K a)^T (y - K a) + lambda a^T K a
          = (K + lambda I)^{-1} y
    note that (K + lambda I)^{-1} is stored in kpinv

    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    y: vector of length n, responses
    alpha: vector of n, coefficients
    */

    /* DSPMV(UPLO,N,ALPHA,AP,X,INCX,BETA,Y,INCY) */
    const char upper_part = 'U';
    const int inc_one = 1;
    const double zero = 0.0, one = 1.0;
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, y, &inc_one,
        &zero, alpha, &inc_one,1);
}




void get_loocv_objective(const int n,
    const double *restrict kpinv,
    const double *restrict alpha,
    double *restrict objective)
{
    /*
    compute the value of leave-one-out cross validated error

    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    alpha: vector of n, coefficients
    objective: scalar, cross validated error
    */

    double obj = 0.0, temp;
    for (int i = 0; i < n; ++i) {
        temp = alpha[i] / kpinv[DIAG_ELEM(i)];
        obj += (temp > 0.0 ? temp : -temp);
    }
    *objective = obj / n;
}




void get_loocv_gradient(const int n, const int p,
    const double *restrict npdx,
    const double *restrict k,
    const double *restrict kpinv,
    const double *restrict alpha,
    const double *restrict y,
    double *work,
    double *restrict gradient)
{
    /*
    compute the gradient of LOOCV objective function

    npdx: matrix of size n (n + 1) / 2 by p, negative pairwise distance in x
    k: matrix of size n by n, design matrix induced by gaussian kernel
    kpinv: matrix of size n by n, inverse of (k + lambda I_n)
    alpha: vector of length n, coefficients
    y: vector of length n, responses
    work: matrix, of size n by (4 + 2 * n)
    gradient: vector of length (p + 1), the gradient of LOOCV value
        with respect to gamma (of length p) and lambda (scalar)
    */

    double temp, temp2;
    double *restrict vec1 = work + n * 0;
    double *restrict vec2 = work + n * 1;
    double *restrict vec3 = work + n * 2;
    double *restrict vec4 = work + n * 3;
    double *restrict mat1 = work + n * 4;
    double *restrict mat2 = work + n * (4 + n);

    /* compute mat1 = kpinv (diag(vec1) + y vec2^T) kpinv,
       where kpinv = (k + lambda I_n)^{-1} */
    for (int i = 0; i < n; ++i) {
        temp = kpinv[DIAG_ELEM(i)];
        temp2 = (alpha[i] > 0.0 ? 1.0 : (alpha[i] < 0.0 ? -1.0 : 0.0));
        vec1[i] = alpha[i] / (temp * temp) * temp2;
        vec2[i] = -1.0 / temp * temp2;
    }

    memset(mat1, 0, n * n * sizeof(double));

    /* DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA) */
    const double zero = 0.0, one = 1.0;
    const int inc_one = 1;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            vec3[i] = vec4[i] = (i <= j ? kpinv[i + (j * (j + 1)) / 2]
                : kpinv[j + (i * (i + 1)) / 2]);
        }
        F77_NAME(dger)(&n, &n, vec1 + j, vec3, &inc_one, vec4, &inc_one,
            mat1, &n);
    }

    /* DSPMV(UPLO,N,ALPHA,AP,X,INCX,BETA,Y,INCY) */
    const char upper_part = 'U';
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, y, &inc_one,
        &zero, vec3, &inc_one,1);
    F77_NAME(dspmv)(&upper_part, &n, &one, kpinv, vec2, &inc_one,
        &zero, vec4, &inc_one,1);

    /* DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA) */
    F77_NAME(dger)(&n, &n, &one, vec3, &inc_one, vec4, &inc_one, mat1, &n);

    /* change mat1 to packed format consistent with k */
    for (int i2 = 0; i2 < n; ++i2) {
        for (int i1 = 0; i1 < i2; ++i1)
            mat2[i1 + (i2 * (i2 + 1)) / 2] =
                mat1[i1 + n * i2] + mat1[i2 + n * i1];
        mat2[DIAG_ELEM(i2)] = mat1[i2 + n * i2];
    }

    /* compute trace((\partial (k + lambda I_n) / \partial gamma_j) mat2) */
    const int nrow_npdx = (n * (n + 1)) / 2;
    int offset_npdx;
    double grd;
    for (int j = 0; j < p; ++j) {
        grd = 0.0;
        offset_npdx = nrow_npdx * j;
        for (int i = 0; i < nrow_npdx; ++i)
            grd += k[i] * npdx[i + offset_npdx] * mat2[i];
        gradient[j] = grd / n;
    }

    /* compute trace((\partial (k + lambda I_n) / \partial lambda) mat2) */
    grd = 0.0;
    for (int i = 0; i < n; ++i)
        grd += mat2[DIAG_ELEM(i)];
    gradient[p] = grd / n;
}




void predict_response(const int n, const int p, const int new_n,
    const double *restrict x,
    const double *restrict alpha,
    const double *restrict gamma,
    const double intercept,
    const double *restrict new_x,
    double *restrict new_yhat)
{
    /*
    predict the responses for new predictors

    x: matrix of size n by p, predictors
    alpha: vector of length n, coefficients
    gamma: vector of length p, scaling factors
    intercept: scalar, sample mean of y
    new_x: matrix of size new_n by p, new predictors
    new_yhat: vector of length new_n, predicted responses
    */

    double value, term;
    for (int k = 0; k < new_n; ++k) {
        value = intercept;
        for (int i = 0; i < n; ++i) {
            term = 0.0;
            for (int j = 0; j < p; ++j) {
                double temp = new_x[k + new_n * j] - x[i + n * j];
                term += gamma[j] * temp * temp;
            }
            value += alpha[i] * exp(-term);
        }
        new_yhat[k] = value;
    }
}




/*********************************/
/* training in multiple datasets */
/*********************************/




void allocate_training_data(const int n, const int p,
    const double percent, training_data *restrict data)
{
    data->n = n;
    data->percent = percent;
    data->intercept = 0.0;
    const int nrow_npdx = (n * (n + 1)) / 2;
    data->x     = (double *)malloc(n * p * sizeof(double));
    data->npdx  = (double *)malloc(nrow_npdx * p * sizeof(double));
    data->k     = (double *)malloc(n * n * sizeof(double));
    data->kpinv = (double *)malloc(n * n * sizeof(double));
    data->alpha = (double *)malloc(n * sizeof(double));
    data->y     = (double *)malloc(n * sizeof(double));
    data->work  = (double *)malloc(n * (4 + 2 * n) * sizeof(double));
}




void free_training_data(training_data *restrict data)
{
    free(data->x);
    free(data->npdx);
    free(data->k);
    free(data->kpinv);
    free(data->alpha);
    free(data->y);
    free(data->work);
}




training_input *allocate_training_input(const int n,
    const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling)
{
    /*
    group: vector, of length n, with values in 0, ... num_groups - 1
    scaling: vector, of length p, for gamma
    */

    /* cut x, y into blocks according to group */
    int *restrict count = (int *)malloc(num_groups * sizeof(int));
    for (int h = 0; h < num_groups; ++h)
        count[h] = 0;
    for (int i = 0; i < n; ++i)
        ++count[group[i]];

    training_data *restrict data = (training_data *)
        malloc(num_groups * sizeof(training_data));
    for (int h = 0; h < num_groups; ++h) {
        allocate_training_data(count[h], p,
            (double)(count[h]) / (double)n, data + h);
        count[h] = 0;
    }

    for (int i = 0; i < n; ++i) {
        const int h = group[i];
        const int i2 = count[h], n2 = data[h].n;
        for (int j = 0; j < p; ++j)
            data[h].x[i2 + n2 * j] = x[i + n * j];
        data[h].y[i2] = y[i];
        ++count[h];
    }

    for (int h = 0; h < num_groups; ++h) {
        transform_x(count[h], p, data[h].x, data[h].npdx);
        center_y(count[h], data[h].y, &(data[h].intercept));
    }

    free(count);

    /* initialize */
    training_input *restrict input = (training_input *)
        malloc(sizeof(training_input));

    input->p = p;
    input->data = data;
    input->num_groups = num_groups;

    input->scaling = (double *)malloc(p * sizeof(double));
    memcpy(input->scaling, scaling, p * sizeof(double));
    input->gamma = (double *)malloc(p * sizeof(double));
    input->lambda = 1.0;
    input->work = (double *)malloc((p + 1) * sizeof(double));

    return input;
}




void free_training_input(training_input *restrict input)
{
    for (int h = 0; h < input->num_groups; ++h)
        free_training_data(input->data + h);
    free(input->data);

    free(input->scaling);
    free(input->gamma);
    free(input->work);

    free(input);
}




double get_aggregate_loocv_objective(const double *param,
    int dim, void *extra)
{
    /*
    param contains scaled gamma and log lambda
    dim should be equal to p + 1
    */

    training_input *restrict input = (training_input *)extra;
    training_data *restrict data = input->data;

    /* maps param to gamma and lambda */
    for (int j = 0; j < input->p; ++j)
        input->gamma[j] = input->scaling[j] * param[j];
    input->lambda = exp(param[input->p]);

    double value = 0.0, temp = 0.0;
    for (int h = 0; h < input->num_groups; ++h) {
        compute_kernel(data[h].n, input->p,
            data[h].npdx, input->gamma, input->lambda,
            data[h].k, data[h].kpinv);
        estimate_alpha(data[h].n,
            data[h].kpinv, data[h].y, data[h].alpha);
        get_loocv_objective(data[h].n,
            data[h].kpinv, data[h].alpha, &temp);

        value += temp * data[h].percent;
    }

    return value;
}




void get_aggregate_loocv_gradient(const double *param,
    int dim, void *extra, double *value)
{
    /*
    param contains scaled gamma and log lambda
    dim should be equal to p + 1
    */

    training_input *restrict input = (training_input *)extra;
    training_data *restrict data = input->data;

    for (int j = 0; j < input->p + 1; ++j)
        value[j] = 0.0;

    for (int h = 0; h < input->num_groups; ++h) {
        /* no need to run compute_kernel and estimate_alpha
           since gradient(param) will only be called
           right after objective(param) */
        get_loocv_gradient(data[h].n, input->p,
            data[h].npdx, data[h].k, data[h].kpinv,
            data[h].alpha, data[h].y,
            data[h].work, input->work);

        for (int j = 0; j < input->p; ++j)
            value[j] += input->work[j] * input->scaling[j]
                * data[h].percent;
        value[input->p] += input->work[input->p] * input->lambda
            * data[h].percent;
    }
}




void train_model(training_input *restrict input)
{
    const int p = input->p;
    double *param = (double *)malloc((p + 1) * sizeof(double));

    /* initial values */

    int *restrict effect = (int *)malloc(p * sizeof(int));
    memset(effect, 0, p * sizeof(int));

    const double initial_param = 0.5;
    const double small_param = initial_param / 10.0 / p;
    for (int j = 0; j < p; ++j)
        param[j] = small_param;    /* for scaled gamma */
    param[p] = 0.0;                /* for log lambda */

    double value, old;
    double best_value = get_aggregate_loocv_objective(param, p + 1, input);
    int best_j;

    double unit = initial_param;
    const int num_rounds = 10;
    for (int round = 1; round <= num_rounds; ++round) {
        best_j = p;
        for (int j = 0; j < p; ++j) {
            old = param[j];
            param[j] = (effect[j] + 1) * unit;
            value = get_aggregate_loocv_objective(param, p + 1, input);
            if (value < best_value) {
                best_value = value;
                best_j = j;
            }
            param[j] = old;
        }

        /* currently unit == initial_param / round */
        if (best_j < p) {
            ++effect[best_j];
            if (round < num_rounds)
                unit = initial_param / (round + 1);
        } else {  /* best_j == p */
            unit = (round > 1 ? initial_param / (round - 1) : 0.0);
        }

        for (int j = 0; j < p; ++j)
            param[j] = (effect[j] ? effect[j] * unit : small_param);

        if (best_j == p) break;
    }

    /* optimization */

    double *lower = (double *)malloc((p + 1) * sizeof(double));
    double *upper = (double *)malloc((p + 1) * sizeof(double));
    for (int j = 0; j < p; ++j) {
        /* for scaled gamma */
        lower[j] = 0.0001 / p;
        upper[j] = 5.0;
    }
    /* for log lambda */
    lower[p] = -10.0;
    upper[p] = 10.0;

    projected_bfgs(param, p + 1, input, get_aggregate_loocv_objective,
        get_aggregate_loocv_gradient, 1000, 1e-8, lower, upper, 0.2,
        &value);
    /* compute again to ensure that gamma, lambda and value are correct */
    value = get_aggregate_loocv_objective(param, p + 1, input);

    free(param);
    free(effect);
    free(lower);
    free(upper);
}




void kernel_train(const int n, const int p, const int num_groups,
    const double *restrict x, const double *restrict y,
    const int *restrict group, const double *restrict scaling,
    double *restrict node, double *restrict alpha,
    int *restrict offset, double *restrict gamma,
    double *restrict intercept)
{
    /*
    wrapper of train_model

    node: vector, of length (n * p)
    alpha: vector, n
    offset: vector, of length (num_groups + 1)
    intercept: vector, num_groups
    gamma: vector, p
    */

    training_input *input = allocate_training_input(n, p, num_groups,
        x, y, group, scaling);
    training_data *restrict data = input->data;

    train_model(input);

    /* save model */
    int cum_n = 0;
    for (int h = 0; h < num_groups; ++h) {
        memcpy(node + cum_n * p, data[h].x, data[h].n * p * sizeof(double));
        memcpy(alpha + cum_n, data[h].alpha, data[h].n * sizeof(double));
        offset[h] = cum_n;
        intercept[h] = data[h].intercept;
        cum_n += data[h].n;
    }
    offset[num_groups] = n;

    memcpy(gamma, input->gamma, p * sizeof(double));

    free_training_input(input);
}




void kernel_predict(const int p, const int num_groups, const int new_n,
    const double *restrict node, const double *restrict alpha,
    const int *restrict offset, const double *restrict gamma,
    const double *restrict intercept, const double *restrict new_x,
    double *restrict new_yhat)
{
    /*
    wrapper of predict_response
    */

    for (int h = 0; h < num_groups; ++h) {
        predict_response(offset[h + 1] - offset[h], p, new_n,
            node + offset[h] * p, alpha + offset[h], gamma, intercept[h],
            new_x, new_yhat + new_n * h);
    }
}



#include <stdlib.h>
#include <math.h>




void get_regrets_from_outcomes(const double *restrict outcomes,
    const unsigned int n, const unsigned int m,
    double *restrict regrets)
{
    /*
    compute regret, which is the difference between the best outcome
    and the current outcome:
    regret_{i, a} = max_{a'} outcome_{i, a'} - outcome_{i, a}

    outcomes: matrix, n by m
    regrets: matrix, n by m
    m: number of treatments
    */

    double *restrict best = (double *)malloc(n * sizeof(double));
    for (unsigned int i = 0; i < n; ++i)
        best[i] = -INFINITY;

    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int i = 0; i < n; ++i) {
            if (outcomes[i + n * a] > best[i])
                best[i] = outcomes[i + n * a];
        }
    }

    for (unsigned int a = 0; a < m; ++a) {
        for (unsigned int i = 0; i < n; ++i)
            regrets[i + n * a] = best[i] - outcomes[i + n * a];
    }

    free(best);
}



#include <stdlib.h>
#include <string.h>




#define SMALL_COUNT 4




void swap_sort(double *restrict vdat,
    unsigned int *restrict idat,
    const unsigned int n)
{
    #define SWAP(j, k) \
        if (vdat[j] > vdat[k]) { \
            double vtemp = vdat[j]; \
            vdat[j] = vdat[k]; vdat[k] = vtemp; \
            int itemp = idat[j]; \
            idat[j] = idat[k]; idat[k] = itemp; \
        }

    switch (n) {
    case 1:
        break;

    case 2:
        SWAP(0, 1);
        break;

    case 3:
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(0, 1);
        break;

    case 4:
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(2, 3);
        SWAP(0, 1);
        SWAP(1, 2);
        SWAP(0, 1);
        break;
    }

    #undef SWAP
}




void combine_arrays(double *restrict vsrc, double *restrict vdst,
    unsigned int *restrict isrc, unsigned int *restrict idst,
    const unsigned int mid, const unsigned int cnt)
{
    unsigned int jl = 0, jr = mid, k = 0;
    for (; k < cnt; ++k) {
        if ((jl < mid) && ((jr >= cnt) || (vsrc[jl] <= vsrc[jr]))) {
            vdst[k] = vsrc[jl];
            idst[k] = isrc[jl];
            ++jl;
        } else {
            vdst[k] = vsrc[jr];
            idst[k] = isrc[jr];
            ++jr;
        }
    }

}




void do_merge_sort(double *restrict vdat, unsigned int *restrict idat,
    double *restrict vbuf, unsigned int *restrict ibuf,
    const unsigned int n)
{
    const unsigned int off2 = n / 2;
    const unsigned int rest = n - off2;
    const unsigned int off1 = off2 / 2;
    const unsigned int off3 = off2 + rest / 2;
    const unsigned int len1 = off2 - off1;
    const unsigned int len2 = off3 - off2;
    const unsigned int len3 = n - off3;

    #define SORT(count, value, index) \
        if (count > SMALL_COUNT) \
            do_merge_sort(value, index, vbuf, ibuf, count); \
        else \
            swap_sort(value, index, count);

    /* sorts four subarrays */
    SORT(off1, vdat,        idat       );
    SORT(len1, vdat + off1, idat + off1);
    SORT(len2, vdat + off2, idat + off2);
    SORT(len3, vdat + off3, idat + off3);

    #undef SORT

    /* checks if array is already ordered */
    if ((vdat[off1 - 1] <= vdat[off1]) &&
        (vdat[off2 - 1] <= vdat[off2]) &&
        (vdat[off3 - 1] <= vdat[off3]))
        return;

    #define COPY(vsrc, vdst, isrc, idst, cnt) \
        memcpy(vdst, vsrc, (cnt) * sizeof(double)); \
        memcpy(idst, isrc, (cnt) * sizeof(unsigned int));

    /* merges array[0..off1) and array[off1..off2) */
    if (vdat[0] > vdat[off2 - 1]) {
        COPY(vdat + off1, vbuf, idat + off1, ibuf, len1);
        COPY(vdat, vbuf + len1, idat, ibuf + len1, off1);
    } else if (vdat[off1 - 1] <= vdat[off1]) {
        COPY(vdat, vbuf, idat, ibuf, off2);
    } else {
        combine_arrays(vdat, vbuf, idat, ibuf, off1, off2);
    }

    /* merges array[off2..off3) and array[off3..n) */
    if (vdat[off2] > vdat[n - 1]) {
        COPY(vdat + off3, vbuf + off2,
            idat + off3, ibuf + off2, len3);
        COPY(vdat + off2, vbuf + off2 + len3,
            idat + off2, ibuf + off2 + len3, len2);
    } else if (vdat[off3 - 1] <= vdat[off3]) {
        COPY(vdat + off2, vbuf + off2,
            idat + off2, ibuf + off2, rest);
    } else {
        combine_arrays(vdat + off2, vbuf + off2,
            idat + off2, ibuf + off2, len2, rest);
    }

    /* merges array[0..off2) and array[off2..n) */
    if (vbuf[0] > vbuf[n - 1]) {
        COPY(vbuf + off2, vdat, ibuf + off2, idat, rest);
        COPY(vbuf, vdat + rest, ibuf, idat + rest, off2);
    } else {
        combine_arrays(vbuf, vdat, ibuf, idat, off2, n);
    }

    #undef COPY
}




void *allocate_space_for_merge_sort(const unsigned int n)
{
    const unsigned int size = n * (sizeof(double) + sizeof(unsigned int));
    return malloc(size);
}




void free_space(void *work)
{
    free(work);
}




void merge_sort(double *restrict vdat,
    unsigned int *restrict idat, unsigned int *restrict rdat,
    const unsigned int n, void *restrict work)
{
    /* ordering indices */
    for (unsigned int j = 0; j < n; ++j)
        idat[j] = j;

    if (n > SMALL_COUNT) {
        do_merge_sort(vdat, idat,
            (double *)work, (unsigned int *)(work + n * sizeof(double)), n);
    } else {
        swap_sort(vdat, idat, n);
    }

    /* ranks (zero-based) */
    #define DELTA 1e-8

    rdat[idat[0]] = 0;
    double last_value = vdat[0];
    unsigned int last_rank = 0;
    for (unsigned int j = 1; j < n; ++j) {
        if (vdat[j] - last_value < DELTA) {
            rdat[idat[j]] = last_rank;
        } else {
            rdat[idat[j]] = j;
            last_value = vdat[j];
            last_rank = j;
        }
    }

    #undef DELTA
}




#undef SMALL_COUNT




/*
    Sort matrix z column by column.
    Obtain ordering indices iz and ranks rz.
    The dimension of z is n by p.
*/

void sort_matrix(const double *restrict z,
    unsigned int *restrict iz,
    unsigned int *restrict rz,
    double *restrict sz,
    const unsigned int n, const unsigned int p)
{
    memcpy(sz, z, n * p * sizeof(double));
    void *work = allocate_space_for_merge_sort(n);

    for (unsigned int j = 0; j < p; ++j) {
        unsigned int offset = n * j;
        merge_sort(sz + offset, iz + offset, rz + offset, n, work);
    }

    free_space(work);
}




void reverse_sort(const unsigned int *restrict iz,
    const unsigned int *restrict rz,
    const double *restrict sz,
    unsigned int *restrict rev_iz,
    unsigned int *restrict rev_rz,
    double *restrict rev_sz,
    const unsigned int n, const unsigned int p)
{
    for (unsigned int j = 0; j < p; ++j) {
        unsigned int offset = n * j;
        for (unsigned int i = 0; i < n; ++i) {
            rev_iz[offset + i] = iz[offset + n - 1 - i];
            rev_rz[offset + i] = n - 1 - rz[offset + i];
            rev_sz[offset + i] = sz[offset + n - 1 - i];
        }
    }
}



#include <stdlib.h>
#include <string.h>
#include <math.h>




typedef struct {
    unsigned int count; /* numer of observations within subtree */
    double total;       /* total sum within subtree */
    unsigned char cut;  /* 0 for left, 1 for right */
    double best;        /* best sum within subtree */
} node;




void minimize_loss(const unsigned int *restrict ix,
    const unsigned int *restrict rx,
    const double *restrict sx,
    const unsigned int *restrict iy,
    const unsigned int *restrict ry,
    const double *restrict sy,
    const double *restrict loss,
    const unsigned int *restrict next,
    const unsigned int n, const double zeta, const double eta,
    double *restrict opt_cx, double *restrict opt_cy,
    double *restrict opt_loss)
{
    /*
    find a, b to minimize
        \sum_{i: x_i <= a, y_i <= b} (loss_i - zeta)
        - eta I(a = Inf) - eta I(b = Inf).
    return the optimal a, b as opt_cx and opt_cy
    and return the minimum in opt_loss.

    sx, ix, rx, sy, iy, ry, loss: vector, of length n
    next: vector, of length (n + 1)
    opt_cx, opt_cy, opt_loss: scalar
    */

    #define DELTA 1e-8

    if ((sx[n - 1] - sx[0] < DELTA) && (sy[n - 1] - sy[0] < DELTA)) {
        /* both x and y consist of constants */
        *opt_cx = sx[0]; *opt_cy = sy[0];
        *opt_loss = INFINITY;
        return;
    }

    #undef DELTA

    unsigned int tree_len = 1, k = 1, succ_k;
    while (k * 2 < n) {
        k *= 2;
        tree_len += k;
    }
    unsigned int array_len = k * 2;

    unsigned int entire_size = (tree_len + array_len) * sizeof(node);
    node *restrict tree = (node *)malloc(entire_size);

    unsigned int i, p, lchild, rchild;
    double left_best, right_best;
    unsigned int best_rx = 0, best_ry = 0;
    double optimum = INFINITY;

    /* set all fields to 0 */
    memset(tree, 0, entire_size);

    /* set the "penalty" for the infinite cutoff of x */
    p = tree_len + n - 1;
    i = 1;
    while (1) {
        tree[p].total = tree[p].best = -eta;
        tree[p].cut = i;
        if (p == 0) break;
        i = (p - 1) % 2;
        p = (p - 1) / 2;
    }

    k = 0;
    while (k < n && next[iy[k]] > iy[k])
        ++k;
    while (k < n) {
        succ_k = k + 1;
        while (succ_k < n && next[iy[succ_k]] > iy[succ_k])
            ++succ_k;

        i = iy[k];    /* definition of k ensures next[i] == i */

        #define EPSILON 1e-10
        #define UPDATE_TREE \
            do { \
                p = (p - 1) / 2; \
                lchild = 2 * p + 1; \
                rchild = lchild + 1; \
                \
                tree[p].count = tree[lchild].count + tree[rchild].count; \
                tree[p].total = tree[lchild].total + tree[rchild].total; \
                left_best = tree[lchild].best; \
                right_best = tree[lchild].total + tree[rchild].best; \
                if (tree[lchild].count == 0 \
                    || right_best < left_best - EPSILON) { \
                    tree[p].best = right_best; \
                    tree[p].cut = 1; \
                } else { \
                    tree[p].best = left_best; \
                    tree[p].cut = 0; \
                } \
            } while (p);

        p = tree_len + rx[i];  /* tied values in x have the same rx */
        tree[p].count += 1;
        tree[p].total += loss[i] - zeta;
        tree[p].best = tree[p].total;
        tree[p].cut = 1;       /* always inclusive at leaf level */
        UPDATE_TREE;           /* update the binary tree */

        if (succ_k >= n) {
            /* set the "penalty" for the infinite cutoff of y */
            p = tree_len;
            while (p < tree_len + n - 1 && tree[p].count == 0)
                ++p;           /* find the first observation */
            tree[p].total -= eta;
            tree[p].best = tree[p].total;
            UPDATE_TREE;
        }

        #undef EPSILON
        #undef UPDATE_TREE

        /* ties in y; note that i == iy[k] */
        if ((succ_k >= n) || (ry[iy[succ_k]] != ry[i])) {
            if (tree[0].best < optimum) {
                optimum = tree[0].best;

                best_rx = 0;
                while (best_rx < tree_len)
                    best_rx = 2 * best_rx + 1 + tree[best_rx].cut;
                best_rx -= tree_len;

                best_ry = k;
            }
        }

        k = succ_k;
    }

    /* just in case something goes wrong */
    if (best_rx >= n) best_rx = n - 1;
    if (best_ry >= n) best_ry = n - 1;

    /* adjust best_rx and best_ry for ties and get cutoff values */
    k = best_rx;
    succ_k = k + 1;
    while (succ_k < n &&
        (rx[ix[succ_k]] == best_rx || next[ix[succ_k]] > ix[succ_k]))
        ++succ_k;

    if (succ_k < n) {
        *opt_cx = 0.5 * (sx[k] + sx[succ_k]);
    } else {
        *opt_cx = (sx[0] <= sx[n - 1] ? INFINITY : -INFINITY);
    }

    k = best_ry;
    succ_k = k + 1;
    while (succ_k < n &&
        (ry[iy[succ_k]] == best_ry || next[iy[succ_k]] > iy[succ_k]))
        ++succ_k;

    if (succ_k < n) {
        *opt_cy = 0.5 * (sy[k] + sy[succ_k]);
    } else {
        *opt_cy = (sy[0] <= sy[n - 1] ? INFINITY : -INFINITY);
    }

    *opt_loss = optimum;

    free(tree);
}



#include <stdlib.h>
#include <string.h>
#include <math.h>




void find_clause(const unsigned int *restrict asc_iz,
    const unsigned int *restrict asc_rz,
    const double *restrict asc_sz,
    const unsigned int *restrict desc_iz,
    const unsigned int *restrict desc_rz,
    const double *restrict desc_sz,
    const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    statement *restrict clause)
{
    /*
    loop over covariate pair and treatment to find the best covariate(s)
    and treatment to use and call minimize_loss to find the best cutoff values

    asc_z, asc_iz, asc_rz: matrix of predictors, of size n by p
    desc_z, desc_iz, desc_rz: matrix of predictors, of size n by p
    regrets: matrix of potential improvements, of size n by m
    next: vector of indices, of length (n + 1),
          specifying the observations to use
    clause: scalar
    */

    double c1, c2, loss, opt_loss = INFINITY;
    int redundant1 = 0;

    /* loop over combinations of covariates */
    for (unsigned int j1 = 0; j1 < p; ++j1) {
        const unsigned int off1 = n * j1;

        #define DELTA 1e-8
        if (asc_sz[n - 1 + off1] - asc_sz[off1] < DELTA) {
            /* z_{j1} is constant */
            if (redundant1) continue;
            redundant1 = 1;
        }
        int redundant2 = 0;

        for (unsigned int j2 = j1 + 1; j2 < p; ++j2) {
            const unsigned int off2 = n * j2;

            if (asc_sz[n - 1 + off2] - asc_sz[off2] < DELTA) {
                /* z_{j2} is constant */
                if (redundant2) continue;
                redundant2 = 1;
            }

            for (unsigned int a = 0; a < m; ++a) {
                /* loop over action options */

                #define EPSILON 1e-10
                #define SEARCH(_type, _iz1, _rz1, _sz1, _iz2, _rz2, _sz2) \
                { \
                    minimize_loss( \
                        _iz1 + off1, _rz1 + off1, _sz1 + off1, \
                        _iz2 + off2, _rz2 + off2, _sz2 + off2, \
                        regrets + n * a, next, \
                        n, zeta, eta, \
                        &c1, &c2, &loss); \
                    if (loss < opt_loss - EPSILON) { \
                        /* use EPSILON to ignore numerical rounding errors */ \
                        opt_loss = loss; \
                        clause->a = a; \
                        clause->type = _type; \
                        clause->j1 = j1; \
                        clause->j2 = j2; \
                        clause->c1 = c1; \
                        clause->c2 = c2; \
                    } \
                }

                SEARCH(CONDITION_TYPE_LL,
                    asc_iz, asc_rz, asc_sz, asc_iz, asc_rz, asc_sz);
                SEARCH(CONDITION_TYPE_LR,
                    asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz);
                SEARCH(CONDITION_TYPE_RL,
                    desc_iz, desc_rz, desc_sz, asc_iz, asc_rz, asc_sz);
                SEARCH(CONDITION_TYPE_RR,
                    desc_iz, desc_rz, desc_sz, desc_iz, desc_rz, desc_sz);

                #undef EPSILON
                #undef SEARCH
            }  /* loop over a */
        }  /* loop over j2 */
        #undef DELTA
    }  /* loop over j1 */
}




void find_last_clause(const double *restrict regrets,
    const unsigned int *restrict next,
    const unsigned int n, const unsigned int m,
    statement *restrict clause)
{
    /*
    for the last clause, no covariate can be used
    only the best treatment needs to be determined

    clause: scalar
    */

    double opt_loss = INFINITY;
    unsigned int opt_a = 0;

    for (unsigned int a = 0; a < m; ++a) {
        double loss = 0.0;
        unsigned int i = next[0], off = n * a;
        while (i < n) {
            loss += regrets[i + off];
            i = next[i + 1];
        }

        if (loss < opt_loss) {
            opt_loss = loss;
            opt_a = a;
        }
    }

    clause->a = opt_a;
    clause->type = CONDITION_TYPE_LL;
    clause->j1 = 0;
    clause->j2 = 1;
    clause->c1 = INFINITY;
    clause->c2 = INFINITY;
}




void apply_clause(const double *restrict z,
    unsigned int *restrict next, const unsigned int n,
    const statement *restrict clause, int *restrict action)
{
    /*
    assign the recommended action dictated by the given clause

    z: matrix of predictors, of size n by p
    next: vector of indices, of length (n + 1),
          specifying the observations to use
    clause: scalar
    action: vector of actions, of length n
            for those who failed the if-condition in the clause,
            their values are not changed
    */

    const double *restrict z1 = z + n * clause->j1;
    const double *restrict z2 = z + n * clause->j2;
    const double c1 = clause->c1, c2 = clause->c2;
    const unsigned int a = clause->a;

    /* update action */
    unsigned int i = next[0];
    switch (clause->type) {
    case CONDITION_TYPE_LL:
        while (i < n) {
            if (z1[i] <= c1 && z2[i] <= c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_LR:
        while (i < n) {
            if (z1[i] <= c1 && z2[i] > c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_RL:
        while (i < n) {
            if (z1[i] > c1 && z2[i] <= c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;

    case CONDITION_TYPE_RR:
        while (i < n) {
            if (z1[i] > c1 && z2[i] > c2)
                action[i] = a;
            i = next[i + 1];
        }
        break;
    }

    /* update next */
    i = n;
    do {
        --i;
        if (action[i] >= 0 || next[i] > i)
            next[i] = next[i + 1];
    } while (i);
}



#include <stdlib.h>
#include <string.h>
#include <math.h>




void find_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double zeta, const double eta,
    const unsigned int max_length,
    statement *restrict rule, unsigned int *restrict rule_length,
    int *restrict action)
{
    /*
    build clauses one by one to form a rule

    z: matrix of predictors, of size n by p
    regrets: matrix of potential improvements, of size n by m
    zeta, eta: tuning parameters
    action: vector, of length n
    rule: vector, of length at least max_length
    rule_length: scalar, the actual length of the rule returned
    */

    /* sort z */
    unsigned int size = n * p * sizeof(double);
    double *restrict asc_sz = (double *)malloc(size);
    double *restrict desc_sz = (double *)malloc(size);

    size = n * p * sizeof(unsigned int);
    unsigned int *restrict asc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict asc_rz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_rz = (unsigned int *)malloc(size);

    sort_matrix(z, asc_iz, asc_rz, asc_sz, n, p);
    reverse_sort(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz, n, p);

    /* initialize next and action */
    unsigned int *restrict next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        next[i] = i;
        action[i] = -1;
    }
    next[n] = n;

    /* build clauses */
    for (unsigned int k = 0; (next[0] < n) && (k + 1 < max_length); ++k) {
        find_clause(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz,
            regrets, next, n, p, m, zeta, eta, rule + k);
        apply_clause(z, next, n, rule + k, action);
        *rule_length = k + 1;
    }

    if (next[0] < n) {
        /* for the rest, defaults to the overall best treatment */
        unsigned int k = max_length - 1;
        find_last_clause(regrets, next, n, m, rule + k);
        apply_clause(z, next, n, rule + k, action);
        *rule_length = max_length;
    }

    /* clean up */
    free(asc_sz);
    free(asc_iz);
    free(asc_rz);
    free(desc_sz);
    free(desc_iz);
    free(desc_rz);

    free(next);
}




void apply_rule(const double *restrict z,
    const unsigned int n,
    const statement *rule, const unsigned int rule_length,
    int *restrict action)
{
    /*
    apply the given rule to each row of z to get an action

    action: vector, of length n
    */

    /* initialize next and action */
    unsigned int *restrict next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    for (unsigned int i = 0; i < n; ++i) {
        next[i] = i;
        action[i] = -1;
    }
    next[n] = n;    /* next is of length (n + 1) */

    /* apply clauses */
    for (unsigned int k = 0; (next[0] < n) && (k < rule_length); ++k) {
        apply_clause(z, next, n, rule + k, action);
    }

    /* clean up */
    free(next);
}




void cv_tune_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const int *restrict fold, const unsigned int num_folds,
    double *restrict cv_regret)
{
    /*
    use cross validation to choose zeta and eta

    fold: vector of length n with values in {0, ..., num_folds - 1}
    zeta_choices, eta_choices: vector of length num_choices
    cv_regret: vector of cross validated mean regret, of length num_choices
    */

    /* sort z */
    unsigned int size = n * p * sizeof(double);
    double *restrict asc_sz = (double *)malloc(size);
    double *restrict desc_sz = (double *)malloc(size);

    size = n * p * sizeof(unsigned int);
    unsigned int *restrict asc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict asc_rz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_rz = (unsigned int *)malloc(size);

    sort_matrix(z, asc_iz, asc_rz, asc_sz, n, p);
    reverse_sort(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz, n, p);

    /* allocate memory */
    unsigned int *restrict train_next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    unsigned int *restrict test_next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    int *restrict train_action = (int *)malloc(n * sizeof(int));
    int *restrict test_action = (int *)malloc(n * sizeof(int));

    statement *restrict rule = (statement *)
        malloc(max_length * sizeof(statement));
    unsigned int rule_length = 0;

    for (unsigned int index_choice = 0; index_choice < num_choices;
        ++index_choice) {
        const double zeta = zeta_choices[index_choice];
        const double eta = eta_choices[index_choice];

        /* initialize test_action */
        for (unsigned int i = 0; i < n; ++i)
            test_action[i] = -1;

        for (unsigned int index_fold = 0; index_fold < num_folds;
            ++index_fold) {
            /* initialize train_next, test_next and train_action */
            train_next[n] = test_next[n] = n;
            unsigned int i = n;
            do {
                --i;
                if (fold[i] == index_fold) {
                    /* this observation goes to the test set */
                    train_next[i] = train_next[i + 1];
                    test_next[i] = i;
                } else {
                    /* this observation goes to the train set */
                    train_next[i] = i;
                    test_next[i] = test_next[i + 1];
                }
            } while (i);

            for (unsigned int i = 0; i < n; ++i)
                train_action[i] = -1;

            /* build clauses */
            for (unsigned int k = 0; (train_next[0] < n)
                && (k + 1 < max_length); ++k) {
                find_clause(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz,
                    regrets, train_next, n, p, m, zeta, eta, rule + k);
                apply_clause(z, train_next, n, rule + k, train_action);
                rule_length = k + 1;
            }

            if (train_next[0] < n) {
                /* for the rest, defaults to the overall best treatment */
                /* and no need to call apply_clause for updating train_next */
                unsigned int k = max_length - 1;
                find_last_clause(regrets, train_next, n, m, rule + k);
                rule_length = max_length;
            }

            /* apply clauses */
            for (unsigned int k = 0; (test_next[0] < n) && (k < rule_length);
                ++k) {
                apply_clause(z, test_next, n, rule + k, test_action);
            }

        } /* loop over index_fold */

        /* evaluate rule using test_action */
        double temp = 0.0;
        for (unsigned int i = 0; i < n; ++i)
            temp += regrets[i + test_action[i] * n];

        cv_regret[index_choice] = temp / n;

    } /* loop over index_choice */

    /* clean up */
    free(asc_sz);
    free(asc_iz);
    free(asc_rz);
    free(desc_sz);
    free(desc_iz);
    free(desc_rz);

    free(train_next);
    free(test_next);
    free(train_action);
    free(test_action);

    free(rule);
}




void batch_evaluate_rule(const double *restrict z,
    const double *restrict regrets,
    const unsigned int n, const unsigned int p, const unsigned int m,
    const double *restrict zeta_choices, const double *restrict eta_choices,
    const unsigned int num_choices,
    const unsigned int max_length,
    const double *restrict test_z,
    const double *restrict test_regrets,
    const unsigned int test_n,
    double *restrict oos_regret)
{
    /*
    use train/test to choose zeta and eta

    oos_regret: vector of out of sample mean regret, of length num_choices
    */

    /* sort z */
    unsigned int size = n * p * sizeof(double);
    double *restrict asc_sz = (double *)malloc(size);
    double *restrict desc_sz = (double *)malloc(size);

    size = n * p * sizeof(unsigned int);
    unsigned int *restrict asc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict asc_rz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_iz = (unsigned int *)malloc(size);
    unsigned int *restrict desc_rz = (unsigned int *)malloc(size);

    sort_matrix(z, asc_iz, asc_rz, asc_sz, n, p);
    reverse_sort(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz, n, p);

    /* allocate memory */
    statement *restrict rule = (statement *)
        malloc(max_length * sizeof(statement));
    unsigned int rule_length = 0;

    unsigned int *restrict next = (unsigned int *)
        malloc((n + 1) * sizeof(unsigned int));
    int *restrict action = (int *)malloc(n * sizeof(int));

    unsigned int *restrict test_next = (unsigned int *)
        malloc((test_n + 1) * sizeof(unsigned int));
    int *restrict test_action = (int *)malloc(test_n * sizeof(int));

    /* find and apply rule */
    for (unsigned int index_choice = 0; index_choice < num_choices;
        ++index_choice) {
        const double zeta = zeta_choices[index_choice];
        const double eta = eta_choices[index_choice];

        /* initialize next and action */
        for (unsigned int i = 0; i < n; ++i) {
            next[i] = i;
            action[i] = -1;
        }
        next[n] = n;

        /* build clauses */
        for (unsigned int k = 0; (next[0] < n) && (k < max_length); ++k) {
            find_clause(asc_iz, asc_rz, asc_sz, desc_iz, desc_rz, desc_sz,
                regrets, next, n, p, m, zeta, eta, rule + k);
            apply_clause(z, next, n, rule + k, action);
            rule_length = k + 1;
        }

        /* initialize test_next and test_action */
        for (unsigned int i = 0; i < test_n; ++i) {
            test_next[i] = i;
            test_action[i] = -1;
        }
        test_next[test_n] = test_n;

        /* apply clauses */
        for (unsigned int k = 0; k < rule_length; ++k) {
            apply_clause(test_z, test_next, test_n, rule + k, test_action);
        }

        /* evaluate rule using test_action */
        double temp = 0.0;
        for (unsigned int i = 0; i < test_n; ++i) {
            temp += test_regrets[i + test_action[i] * test_n];
        }
        oos_regret[index_choice] = temp / test_n;
    } /* loop over index_choice */

    /* clean up */
    free(asc_sz);
    free(asc_iz);
    free(asc_rz);
    free(desc_sz);
    free(desc_iz);
    free(desc_rz);

    free(next);
    free(action);
    free(test_next);
    free(test_action);

    free(rule);
}



#include <R.h>
#include <Rinternals.h>




SEXP R_kernel_train(SEXP R_x, SEXP R_y, SEXP R_group, SEXP R_num_groups,
    SEXP R_scaling)
{
    const int n = nrows(R_x), p = ncols(R_x);
    const int num_groups = INTEGER(R_num_groups)[0];

    SEXP R_list;
    PROTECT(R_list = allocVector(VECSXP, 5));

    SEXP R_node, R_alpha, R_offset, R_gamma, R_intercept;
    PROTECT(R_node = allocVector(REALSXP, n * p));
    PROTECT(R_alpha = allocVector(REALSXP, n));
    PROTECT(R_offset = allocVector(INTSXP, num_groups + 1));
    PROTECT(R_gamma = allocVector(REALSXP, p));
    PROTECT(R_intercept = allocVector(REALSXP, num_groups));
    SET_VECTOR_ELT(R_list, 0, R_node);
    SET_VECTOR_ELT(R_list, 1, R_alpha);
    SET_VECTOR_ELT(R_list, 2, R_offset);
    SET_VECTOR_ELT(R_list, 3, R_gamma);
    SET_VECTOR_ELT(R_list, 4, R_intercept);

    kernel_train(n, p, num_groups,
        REAL(R_x), REAL(R_y),
        INTEGER(R_group), REAL(R_scaling),
        REAL(R_node), REAL(R_alpha), INTEGER(R_offset),
        REAL(R_gamma), REAL(R_intercept));

    SEXP R_names;
    PROTECT(R_names = allocVector(STRSXP, 5));
    SET_STRING_ELT(R_names, 0, mkChar("node"));
    SET_STRING_ELT(R_names, 1, mkChar("alpha"));
    SET_STRING_ELT(R_names, 2, mkChar("offset"));
    SET_STRING_ELT(R_names, 3, mkChar("gamma"));
    SET_STRING_ELT(R_names, 4, mkChar("intercept"));
    namesgets(R_list, R_names);

    UNPROTECT(7);  /* 1 + 5 + 1 */
    return R_list;
}




SEXP R_kernel_predict(SEXP R_list, SEXP R_new_x)
{
    SEXP R_node, R_alpha, R_offset, R_gamma, R_intercept;

    R_node      = VECTOR_ELT(R_list, 0);
    R_alpha     = VECTOR_ELT(R_list, 1);
    R_offset    = VECTOR_ELT(R_list, 2);
    R_gamma     = VECTOR_ELT(R_list, 3);
    R_intercept = VECTOR_ELT(R_list, 4);

    const int p = length(R_gamma);
    const int num_groups = length(R_intercept);
    const int new_n = nrows(R_new_x);

    SEXP R_new_yhat;
    PROTECT(R_new_yhat = allocMatrix(REALSXP, new_n, num_groups));

    kernel_predict(p, num_groups, new_n,
        REAL(R_node), REAL(R_alpha), INTEGER(R_offset),
        REAL(R_gamma), REAL(R_intercept),
        REAL(R_new_x), REAL(R_new_yhat));

    UNPROTECT(1);
    return R_new_yhat;
}




SEXP R_get_regrets_from_outcomes(SEXP R_outcomes)
{
    const int n = nrows(R_outcomes);
    const int num_groups = ncols(R_outcomes);

    SEXP R_regrets;
    PROTECT(R_regrets = allocMatrix(REALSXP, n, num_groups));

    get_regrets_from_outcomes(REAL(R_outcomes), n, num_groups,
        REAL(R_regrets));

    UNPROTECT(1);
    return R_regrets;
}




SEXP R_save_rule_as_list(const statement *restrict rule,
    const unsigned int rule_length)
{
    SEXP R_list;
    PROTECT(R_list = allocVector(VECSXP, 6));

    SEXP R_a, R_type, R_j1, R_j2, R_c1, R_c2;
    PROTECT(R_a = allocVector(INTSXP, rule_length));
    PROTECT(R_type = allocVector(STRSXP, rule_length));
    PROTECT(R_j1 = allocVector(INTSXP, rule_length));
    PROTECT(R_j2 = allocVector(INTSXP, rule_length));
    PROTECT(R_c1 = allocVector(REALSXP, rule_length));
    PROTECT(R_c2 = allocVector(REALSXP, rule_length));
    SET_VECTOR_ELT(R_list, 0, R_a);
    SET_VECTOR_ELT(R_list, 1, R_type);
    SET_VECTOR_ELT(R_list, 2, R_j1);
    SET_VECTOR_ELT(R_list, 3, R_j2);
    SET_VECTOR_ELT(R_list, 4, R_c1);
    SET_VECTOR_ELT(R_list, 5, R_c2);

    for (unsigned int i = 0; i < rule_length; ++i) {
        INTEGER(R_a)[i] = rule[i].a;
        switch (rule[i].type) {
        case CONDITION_TYPE_LL:
            SET_STRING_ELT(R_type, i, mkChar("LL"));
            break;
        case CONDITION_TYPE_LR:
            SET_STRING_ELT(R_type, i, mkChar("LR"));
            break;
        case CONDITION_TYPE_RL:
            SET_STRING_ELT(R_type, i, mkChar("RL"));
            break;
        case CONDITION_TYPE_RR:
            SET_STRING_ELT(R_type, i, mkChar("RR"));
            break;
        default:
            SET_STRING_ELT(R_type, i, mkChar("--"));
        }
        INTEGER(R_j1)[i] = rule[i].j1 + 1;
        INTEGER(R_j2)[i] = rule[i].j2 + 1;
        REAL(R_c1)[i] = rule[i].c1;
        REAL(R_c2)[i] = rule[i].c2;
    }

    SEXP R_names, R_rows, R_class;
    PROTECT(R_names = allocVector(STRSXP, 6));
    SET_STRING_ELT(R_names, 0, mkChar("a"));
    SET_STRING_ELT(R_names, 1, mkChar("type"));
    SET_STRING_ELT(R_names, 2, mkChar("j1"));
    SET_STRING_ELT(R_names, 3, mkChar("j2"));
    SET_STRING_ELT(R_names, 4, mkChar("c1"));
    SET_STRING_ELT(R_names, 5, mkChar("c2"));
    namesgets(R_list, R_names);

    char buffer[255];
    PROTECT(R_rows = allocVector(STRSXP, rule_length));
    for (unsigned int i = 0; i < rule_length; ++i) {
        snprintf(buffer, 255, "%d", i + 1);
        SET_STRING_ELT(R_rows, i, mkChar(buffer));
    }
    setAttrib(R_list, R_RowNamesSymbol, R_rows);

    PROTECT(R_class = allocVector(STRSXP, 1));
    SET_STRING_ELT(R_class, 0, mkChar("data.frame"));
    setAttrib(R_list, R_ClassSymbol, R_class);

    UNPROTECT(10);  /* 1 + 6 + 3 */
    return R_list;
}




statement *R_change_list_into_rule(SEXP R_list, unsigned int *rule_length)
{
    SEXP R_a, R_type, R_j1, R_j2, R_c1, R_c2;
    R_a    = VECTOR_ELT(R_list, 0);
    R_type = VECTOR_ELT(R_list, 1);
    R_j1   = VECTOR_ELT(R_list, 2);
    R_j2   = VECTOR_ELT(R_list, 3);
    R_c1   = VECTOR_ELT(R_list, 4);
    R_c2   = VECTOR_ELT(R_list, 5);

    statement *rule = (statement *)malloc(length(R_a) * sizeof(statement));
    for (unsigned int i = 0; i < length(R_a); ++i) {
        rule[i].a = INTEGER(R_a)[i];
        const char *type = CHAR(STRING_ELT(R_type, i));
        if (type[0] == 'L') {
            rule[i].type = (type[1] == 'L' ? CONDITION_TYPE_LL :
                CONDITION_TYPE_LR);
        } else {
            rule[i].type = (type[1] == 'L' ? CONDITION_TYPE_RL :
                CONDITION_TYPE_RR);
        }
        rule[i].j1 = INTEGER(R_j1)[i] - 1;
        rule[i].j2 = INTEGER(R_j2)[i] - 1;
        rule[i].c1 = REAL(R_c1)[i];
        rule[i].c2 = REAL(R_c2)[i];
    }

    *rule_length = length(R_a);
    return rule;
}




SEXP R_find_rule(SEXP R_z, SEXP R_regrets, SEXP R_zeta, SEXP R_eta,
    SEXP R_max_length, SEXP R_action)
{
    const unsigned int max_length = INTEGER(R_max_length)[0];
    unsigned int rule_length = 0;
    statement *rule = (statement *)malloc(max_length * sizeof(statement));

    find_rule(REAL(R_z), REAL(R_regrets),
        nrows(R_z), ncols(R_z), ncols(R_regrets),
        REAL(R_zeta)[0], REAL(R_eta)[0], max_length,
        rule, &rule_length, INTEGER(R_action));
    SEXP R_list = R_save_rule_as_list(rule, rule_length);

    free(rule);
    return R_list;
}




SEXP R_apply_rule(SEXP R_list, SEXP R_z)
{
    const unsigned int n = nrows(R_z);
    SEXP R_action;
    PROTECT(R_action = allocVector(INTSXP, n));

    unsigned int rule_length = 0;
    statement *rule = R_change_list_into_rule(R_list, &rule_length);

    apply_rule(REAL(R_z), n, rule, rule_length, INTEGER(R_action));

    free(rule);

    UNPROTECT(1);
    return R_action;
}




SEXP R_cv_tune_rule(SEXP R_z, SEXP R_regrets,
    SEXP R_zeta_choices, SEXP R_eta_choices,
    SEXP R_max_length,
    SEXP R_fold, SEXP R_num_folds)
{
    const unsigned int num_choices = length(R_zeta_choices);

    SEXP R_cv_regret;
    PROTECT(R_cv_regret = allocVector(REALSXP, num_choices));

    cv_tune_rule(REAL(R_z), REAL(R_regrets),
        nrows(R_z), ncols(R_z), ncols(R_regrets),
        REAL(R_zeta_choices), REAL(R_eta_choices), num_choices,
        INTEGER(R_max_length)[0],
        INTEGER(R_fold), INTEGER(R_num_folds)[0],
        REAL(R_cv_regret));

    UNPROTECT(1);
    return R_cv_regret;
}




SEXP R_batch_evaluate_rule(SEXP R_z, SEXP R_regrets,
    SEXP R_zeta_choices, SEXP R_eta_choices,
    SEXP R_max_length,
    SEXP R_test_z, SEXP R_test_regrets)
{
    const unsigned int num_choices = length(R_zeta_choices);

    SEXP R_oos_regret;
    PROTECT(R_oos_regret = allocVector(REALSXP, num_choices));

    batch_evaluate_rule(REAL(R_z), REAL(R_regrets),
        nrows(R_z), ncols(R_z), ncols(R_regrets),
        REAL(R_zeta_choices), REAL(R_eta_choices), num_choices,
        INTEGER(R_max_length)[0],
        REAL(R_test_z), REAL(R_test_regrets), nrows(R_test_z),
        REAL(R_oos_regret));

    UNPROTECT(1);
    return R_oos_regret;
}




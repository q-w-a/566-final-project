
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <R.h>
#include <Rdefines.h>
#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static struct svm_node ** sparsify (double *x, int r, int c)
{
    struct svm_node** sparse;
    int         i, ii, count;

    sparse = (struct svm_node **) malloc (r * sizeof(struct svm_node *));
    for (i = 0; i < r; i++) {
	/* determine nr. of non-zero elements */
	for (count = ii = 0; ii < c; ii++)
	    if (x[i * c + ii] != 0) count++;

	/* allocate memory for column elements */
	sparse[i] = (struct svm_node *) malloc ((count + 1) * sizeof(struct svm_node));

	/* set column elements */
	for (count = ii = 0; ii < c; ii++)
	    if (x[i * c + ii] != 0) {
		sparse[i][count].index = ii + 1;
		sparse[i][count].value = x[i * c + ii];
		count++;
	    }

	/* set termination element */
	sparse[i][count].index = -1;
    }

    return sparse;
}

static struct svm_node ** transsparse (double *x, int r, int *rowindex, int *colindex)
{
    struct svm_node** sparse;
    int i, ii, count = 0, nnz = 0;

    sparse = (struct svm_node **) malloc (r * sizeof(struct svm_node*));
    for (i = 0; i < r; i++) {
	/* allocate memory for column elements */
	nnz = rowindex[i+1] - rowindex[i];
	sparse[i] = (struct svm_node *) malloc ((nnz + 1) * sizeof(struct svm_node));

	/* set column elements */
	for (ii = 0; ii < nnz; ii++) {
	    sparse[i][ii].index = colindex[count];
	    sparse[i][ii].value = x[count];
	    count++;
	}

	/* set termination element */
	sparse[i][ii].index = -1;
    }

    return sparse;

}

void wsvmtrain (double *x, int *r, int *c,
	       double *y, double *sample_weights,
	       int    *rowindex, int *colindex,
	       int    *svm_type,
	       int    *kernel_type,
	       int    *degree,
	       double *gamma,
	       double *coef0,
	       double *cost,
	       double *nu,
	       int    *weightlabels,
	       double *weights,
	       int    *nweights,
	       double *cache,
	       double *tolerance,
	       double *epsilon,
	       int    *shrinking,
	       int    *sparse,
	       int    *probability,

	       int    *nclasses,
	       int    *nr,
	       int    *index,
	       int    *labels,
	       int    *nSV,
	       double *rho,
	       double *coefs,
	       double *sigma,
	       double *probA,
	       double *probB,

	       char   **error)
{
    struct svm_parameter par;
    struct svm_problem   prob;
    struct svm_model    *model = NULL;
    int i;  /* int i, ii; */
    const char* s;

    /* set parameters */
    par.svm_type    = *svm_type;
    par.kernel_type = *kernel_type;
    par.degree      = *degree;
    par.gamma       = *gamma;
    par.coef0       = *coef0;
    par.cache_size  = *cache;
    par.eps         = *tolerance;
    par.C           = *cost;
    par.nu          = *nu;
    par.nr_weight   = *nweights;
    if (par.nr_weight > 0) {
	par.weight      = (double *) malloc (sizeof(double) * par.nr_weight);
	memcpy(par.weight, weights, par.nr_weight * sizeof(double));
	par.weight_label = (int *) malloc (sizeof(int) * par.nr_weight);
	memcpy(par.weight_label, weightlabels, par.nr_weight * sizeof(int));
    }
    par.p           = *epsilon;
    par.shrinking   = *shrinking;
    par.probability = *probability;

    /* set problem */
    prob.l = *r;
    prob.y = y;
    prob.W = sample_weights;

    if (*sparse > 0)
	prob.x = transsparse(x, *r, rowindex, colindex);
    else
	prob.x = sparsify(x, *r, *c);

    /* check parameters & copy error message */
    s = svm_check_parameter(&prob, &par);
    if (s) {
        strcpy(*error, s);
    } else {

        /* call svm_train */
        model = svm_train(&prob, &par);

        /* set up return values */

        /*	for (ii = 0; ii < model->l; ii++)
            for (i = 0; i < *r;	i++)
            if (prob.x[i] == model->SV[ii]) index[ii] = i+1; */
        svm_get_sv_indices(model, index);

        *nr  = model->l;
        *nclasses = model->nr_class;
        memcpy (rho, model->rho, *nclasses * (*nclasses - 1)/2 * sizeof(double));

        if (*probability && par.svm_type != ONE_CLASS) {
          if (par.svm_type == EPSILON_SVR || par.svm_type == NU_SVR)
            *sigma = svm_get_svr_probability(model);
          else {
            memcpy(probA, model->probA,
                *nclasses * (*nclasses - 1)/2 * sizeof(double));
            memcpy(probB, model->probB,
                *nclasses * (*nclasses - 1)/2 * sizeof(double));
          }
        }

        for (i = 0; i < *nclasses-1; i++)
            memcpy (coefs + i * *nr, model->sv_coef[i],  *nr * sizeof (double));

        if (*svm_type < 2) {
            memcpy (labels, model->label, *nclasses * sizeof(int));
            memcpy (nSV, model->nSV, *nclasses * sizeof(int));
        }

        /* clean up memory */
        svm_free_and_destroy_model(&model);
    }

    /* clean up memory */
    if (par.nr_weight > 0) {
        free(par.weight);
        free(par.weight_label);
    }

    for (i = 0; i < *r; i++) free (prob.x[i]);
        free (prob.x);
}

void wsvmpredict  (int    *decisionvalues,
		  int    *probability,

		  double *v, int *r, int *c,
		  int    *rowindex,
		  int    *colindex,
		  double *coefs,
		  double *rho,
		  int    *compprob,
		  double *probA,
		  double *probB,
		  int    *nclasses,
		  int    *totnSV,
		  int    *labels,
		  int    *nSV,
		  int    *sparsemodel,

		  int    *svm_type,
		  int    *kernel_type,
		  int    *degree,
		  double *gamma,
		  double *coef0,

		  double *x, int *xr,
		  int    *xrowindex,
		  int    *xcolindex,
		  int    *sparsex,

		  double *ret,
		  double *dec,
		  double *prob)
{
    struct svm_model m;
    struct svm_node ** train;
    int i;

    /* set up model */
    m.l        = *totnSV;
    m.nr_class = *nclasses;
    m.sv_coef  = (double **) malloc (m.nr_class * sizeof(double*));
    for (i = 0; i < m.nr_class - 1; i++) {
      m.sv_coef[i] = (double *) malloc (m.l * sizeof (double));
      memcpy (m.sv_coef[i], coefs + i*m.l, m.l * sizeof (double));
    }

    if (*sparsemodel > 0)
	m.SV   = transsparse(v, *r, rowindex, colindex);
    else
	m.SV   = sparsify(v, *r, *c);

    m.rho      = rho;
    m.probA    = probA;
    m.probB    = probB;
    m.label    = labels;
    m.nSV      = nSV;

    /* set up parameter */
    m.param.svm_type    = *svm_type;
    m.param.kernel_type = *kernel_type;
    m.param.degree      = *degree;
    m.param.gamma       = *gamma;
    m.param.coef0       = *coef0;
    m.param.probability = *compprob;

    m.free_sv           = 1;

    /* create sparse training matrix */
    if (*sparsex > 0)
	train = transsparse(x, *xr, xrowindex, xcolindex);
    else
	train = sparsify(x, *xr, *c);

    /* call svm-predict-function for each x-row, possibly using probability
       estimator, if requested */
    if (*probability && svm_check_probability_model(&m)) {
      for (i = 0; i < *xr; i++)
	ret[i] = svm_predict_probability(&m, train[i], prob + i * *nclasses);
    } else {
      for (i = 0; i < *xr; i++)
	ret[i] = svm_predict(&m, train[i]);
    }

    /* optionally, compute decision values */
    if (*decisionvalues)
      for (i = 0; i < *xr; i++)
	svm_predict_values(&m, train[i], dec + i * *nclasses * (*nclasses - 1) / 2);

    /* clean up memory */
    for (i = 0; i < *xr; i++)
	free (train[i]);
    free (train);

    for (i = 0; i < *r; i++)
	free (m.SV[i]);
    free (m.SV);

    for (i = 0; i < m.nr_class - 1; i++)
      free(m.sv_coef[i]);
    free(m.sv_coef);
}

void wsvmwrite (double *v, int *r, int *c,
		  int    *rowindex,
		  int    *colindex,
		  double *coefs,
		  double *rho,
	          int    *compprob,
	          double *probA,
	          double *probB,
		  int    *nclasses,
		  int    *totnSV,
		  int    *labels,
		  int    *nSV,
		  int    *sparsemodel,

		  int    *svm_type,
		  int    *kernel_type,
		  int    *degree,
		  double *gamma,
		  double *coef0,

		  char **filename)

{
    struct svm_model m;
    int i;
    char *fname = *filename;

    /* set up model */
    m.l        = *totnSV;
    m.nr_class = *nclasses;
    m.sv_coef  = (double **) malloc (m.nr_class * sizeof(double*));
    for (i = 0; i < m.nr_class - 1; i++) {
	m.sv_coef[i] = (double *) malloc (m.l * sizeof (double));
	memcpy (m.sv_coef[i], coefs + i*m.l, m.l * sizeof (double));
    }

    if (*sparsemodel > 0)
	m.SV   = transsparse(v, *r, rowindex, colindex);
    else
	m.SV   = sparsify(v, *r, *c);

    m.rho      = rho;
    m.label    = labels;
    m.nSV      = nSV;
    if (*compprob) {
	m.probA    = probA;
	m.probB    = probB;
    } else {
	m.probA    = NULL;
	m.probB    = NULL;
    }

    /* set up parameter */
    m.param.svm_type    = *svm_type;
    m.param.kernel_type = *kernel_type;
    m.param.degree      = *degree;
    m.param.gamma       = *gamma;
    m.param.coef0       = *coef0;

    m.free_sv           = 1;

    /* write svm model */
    svm_save_model(fname, &m);

    for (i = 0; i < m.nr_class - 1; i++)
	free(m.sv_coef[i]);
    free(m.sv_coef);

    for (i = 0; i < *r; i++)
	free (m.SV[i]);
    free (m.SV);

}



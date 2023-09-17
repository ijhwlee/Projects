# ifndef __ERRORFUNCTION__H
# define __ERRORFUNCTION__H

extern double abs_function(int d,int n,double **origy, double** orisigma, double **gdfy);
extern double max_function(int d,int n,double **origy, double** orisigma,double **gdfy);
extern double mse_function(int d,int n,double **origy, double** orisigma,double **gdfy);
extern double logLikelihood(int d, int n, double** origy, double** orisigma, double** gdfy);
extern double user_function(int d,int n,double **origy, double** orisigma,double **gdfy);

# endif

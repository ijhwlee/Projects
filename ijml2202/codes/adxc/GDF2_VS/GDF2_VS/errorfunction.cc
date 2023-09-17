# include <math.h>

double abs_function(int d,int n,double **origy, double** orisigma, double **gdfy)
{
	int i;
	double sum = 0.0;
	for(i=0;i<n;i++)
	{
		double s=0.0;
		for(int j=0;j<d;j++)
			s=s+pow((origy[i][j]-gdfy[i][j])/ orisigma[i][j],2.0);
		s=sqrt(s);
		sum = sum +s;
	}
	return sum/n;
}

double max_function(int d,int n,double **origy, double** orisigma, double **gdfy)
{
	int i;
	double mmax=0;
	for(i=0;i<n;i++)
	{
		double s=0.0;
		for(int j=0;j<d;j++)
			s = s + pow((origy[i][j] - gdfy[i][j]) / orisigma[i][j], 2.0);
		s=sqrt(s);
		if(s>mmax) mmax=s;
	}
	return mmax;
}

// d : nclass, n : tcount
double mse_function(int d,int n,double **origy,double **orisigma, double **gdfy)
{
	int i;
	double sum = 0.0;
	for(i=0;i<n;i++)
	{
		double s=0.0;
		for(int j=0;j<d;j++)
			s = s + pow((origy[i][j] - gdfy[i][j]) / orisigma[i][j], 2.0);
		sum = sum +s;
	}
	return sum/n;
}

// d : nclass, n : tcount
double logLikelihood(int d, int n, double** origy, double** orisigma, double** gdfy)
{
	int i;
	double sum = 0.0;
	for (i = 0; i < n; i++)
	{
		double s = 0.0;
		for (int j = 0; j < d; j++)
			s = s + pow((origy[i][j] - gdfy[i][j]) / orisigma[i][j], 2.0);
		sum = sum + s;
	}
	return sum / 2;
}

double user_function(int d,int n,double **origy, double** orisigma, double **gdfy)
{
	return 0.0;
}

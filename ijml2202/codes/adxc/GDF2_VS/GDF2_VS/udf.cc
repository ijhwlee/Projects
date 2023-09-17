# include <math.h>


double udf1(double x)
{
	//return 1.0/(1.0+exp(-x));
	return x;
}

double udf2(double x)
{
	//return x>0?1:-1;
	return x * x;
}

double udf3(double x)
{
	//return exp(-x*x/2.0);
	return x * x * x;
}
double udf4(double x)
{
	//return 1.0/(1.0+exp(-x));
	return pow(x, 4);
}

double udf5(double x)
{
	//return x>0?1:-1;
	return pow(x, 5);
}

double udf6(double x)
{
	//return exp(-x*x/2.0);
	return pow(x, 6);
}

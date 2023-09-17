# ifndef __DOUBLESTACK__H

#if defined (__SVR4) && defined (__sun)
	# include  <ieeefp.h>
	extern int isinf(double); 
#endif

/* A stack implementation for doubles */
class DoubleStack
{
	private:
		double *data;
		int counter;
	public:
		DoubleStack();
		int  size() const;
		void push(double x);
		double  top() const;
		double pop();
		void	clear();
		~DoubleStack();
};

# define __DOUBLESTACK__H
# endif

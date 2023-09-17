#include "doublestack.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;

#if defined (__SVR4) && defined (__sun)
int isinf(double x) { return !finite(x) && x==x; } 
# endif

/* Implementation of a stack for double */
DoubleStack::DoubleStack()
{
	/* default size of stack: 512 elements */
	data = (double*)malloc(512*sizeof(double));//NULL;
	counter =0;
}

/* empties the stack */
void	DoubleStack::clear()
{
	counter = 0;
}

/* returns the size of the stack */
int	DoubleStack::size() const
{
	return counter;
}

/* pushes an element to the stack */
void DoubleStack::push(double x)
{
	if(isnan(x) || isinf(x)) {x=1e+8; }
	/* if the stack is full: grow the data buffer by one */
	if(counter>=512) 
	{
		data=(double*)realloc(data,(counter+1)*sizeof(double));
	}
	data[counter]=x;
	counter++;
}

double  DoubleStack::top() const
{
	return (counter!=0)?data[counter-1]:-1;
}

/* pops an element from the stack */
double  DoubleStack::pop()
{
	if(!counter) return -1;
	double t=data[counter-1];
	counter--;
	return t;
}

DoubleStack::~DoubleStack()
{
	free(data);
}

#ifndef __GDFPROGRAM__H
#define __GDFPROGRAM__H
#include "program.h"
#include "cprogram.h"
class GdfProgram :
	public Program
{
	private:
		int	dimension;
		Cprogram	*program;
		double		**train_xpoint;
		vector<double>  train_ypoint;
		vector<double>  train_sigma; // added by hwlee, now every data line is x1, x2, ..., xD, y, sigma
		double		**test_xpoint;
		vector<double>	test_ypoint;
		vector<double>	test_sigma;
		double		**gdfy,**tempy, **tempsigma;
		int traincount,testcount;
		double lower_sigma;
		double upper_sigma;
		double check_lower, check_upper, check_delta;
		double lower_value, upper_value; /* lower and upper values for H(z)*/
	public:
		//public method declaration
		GdfProgram(char *GrammarFile,char *TrainFile,char *TestFile);
		virtual double fitness(vector<int> &genome);
		int checkSingularity();
		double	getTrainError(vector<int> &genome);
		double  getTestError(vector<int>  &genome);
		double	getClassError(vector<int> &genome);
		double  getTestError(vector<int>  &genome,char *filename);
		string  printProgram(vector<int>  &genome);
		int	getTrainSize() const;
		int	getTestSize() const;
		~GdfProgram();
};
#endif

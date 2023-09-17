#include "gdfprogram.h"
#include "get_options.h"
#include "errorfunction.h"
#include <math.h>
#include <stdio.h>

static int imax(int a,int b)
{
	return a>b?a:b;
}

/* Constructor */
GdfProgram::GdfProgram(char *GrammarFile,char *TrainFile,char *TestFile)
{
	traincount = 0;
	testcount  = 0;
	/* Read the train file: the first line must be the dimension and the second line the number of examples */
	if(TrainFile!=NULL)
	{
		/*	Estimate the nclass parameter from the file if and
		 *	only if the nclass is equal to 1.
		 * */
		//int d;
		int tcount;
		double dread;
		int    iread=0;
		FILE *fp;
		if(nclass==1)
		{
			fp=fopen(TrainFile,"r");
			if (!fp) {
				printf("[ERROR]Train file(%s) does not exist, stop.\n", TrainFile);
				exit(-2);
				return;
			}
			fscanf(fp,"%d",&dimension); // dimension is dimension of observation data point
			fscanf(fp,"%d",&tcount); // tcount is the number of observation points
			if(tcount<=0) {fclose(fp); return ;}
			while(!feof(fp))
			{
				int d=fscanf(fp,"%lf",&dread);
				if(d<=0) break;
				iread++;
			}
			iread = iread - dimension * tcount;
			iread = iread / tcount;
			nclass = iread/2; // every class data has two columns one for data other for sigma
			fclose(fp);
		}
		fp=fopen(TrainFile,"r");
		fscanf(fp,"%d",&dimension);
		fscanf(fp,"%d",&tcount);
		if(tcount<=0) {fclose(fp); return ;}
		train_xpoint=new double*[tcount];
		train_ypoint.resize(nclass * tcount);
		train_sigma.resize(nclass * tcount);
		traincount = tcount;
		for(int i=0;i<tcount;i++)
		{
			train_xpoint[i]=new double[dimension];
			for(int j=0;j<dimension;j++) 
				fscanf(fp,"%lf",&train_xpoint[i][j]);
			for (int j = 0; j < nclass; j++)
			{
				fscanf(fp, "%lf", &train_ypoint[i * nclass + j]); // read data
				fscanf(fp, "%lf", &train_sigma[i * nclass + j]); // read sigma
			}
		}
		fclose(fp);
	}
	else
		return ;
	/* Read the test file: the first line must be the dimension and the second line the number of examples */
	if(strlen(TestFile))
	{
		FILE *fp=fopen(TestFile,"r");
		if (!fp) {
			printf("[ERROR]Problem file %s is not exist, stopping...\n", TestFile);
			exit(-1);
			return;
		}
		int d;
		fscanf(fp,"%d",&d);
		if(d!=dimension) {fclose(fp); return ;}
		int tcount;
		fscanf(fp,"%d",&tcount);
		if(tcount<=0) {fclose(fp); return ;}
		test_xpoint=new double*[tcount];
		test_ypoint.resize(nclass * tcount);
		test_sigma.resize(nclass * tcount);
		testcount = tcount;
		for(int i=0;i<tcount;i++)
		{
			test_xpoint[i]=new double[dimension];
			for(int j=0;j<dimension;j++)
				fscanf(fp,"%lf",&test_xpoint[i][j]);
			for (int j = 0; j < nclass; j++) {
				fscanf(fp, "%lf", &test_ypoint[i * nclass + j]);
				fscanf(fp, "%lf", &test_sigma[i * nclass + j]);
			}
		}
		fclose(fp);
	}
	int mcount =imax(traincount,testcount);
	gdfy=new double*[mcount];
	tempy=new double*[mcount];
	tempsigma = new double* [mcount];
	for(int i=0;i<mcount;i++)
	{
		gdfy[i]=new double[nclass];
		tempy[i]=new double[nclass];
		tempsigma[i] = new double[nclass];
	}
/* Read the grammar file */
	if(GrammarFile!=NULL)
	{
		program =new Cprogram(GrammarFile,dimension);
		setStartSymbol(program->getStartSymbol());
		this->check_lower = 0.9; /* this is only for LCDM H(z) model, this should be generalized later*/
		this->check_upper = 1.6; /* singularity check variable range */
		this->check_delta = 1.0e-5;
		this->lower_value = 30; /* arbitrarily chosen limits */
		this->upper_value = 600;
		this->lower_sigma = 10;
		this->upper_sigma = 10;
		//this->boundary_type = boundary_type;
	}
	else
		return;
}

/* Return the train error for a particular chromosome (using the train data) */
double	GdfProgram::getTrainError(vector<int> &genome) 
{
	double value=0.0;
	value=-fitness(genome);
	return value;
}

string	GdfProgram::printProgram(vector<int> &genome)
{
	string totalstring="";
	vector<int> pgenome;
	vector<string> pstring;
	double value=0.0;
	pgenome.resize(genome.size()/nclass);
	pstring.resize(nclass);
	for(int n=0;n<nclass;n++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[n*genome.size()/nclass+j];
		int redo=0;
		pstring[n]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) 
		{
			return "";
		}
		char str[100];
		sprintf(str,"f%d(x)= ",n+1);
		totalstring+=str+pstring[n]+"\n";
	}
	return totalstring;
}

/* Return the test error for a particular chromosome (using the test data) */
double	GdfProgram::getTestError(vector<int> &genome,char *filename) 
{
	vector<int> pgenome;
	vector<string> pstring;
	double value=0.0;
	pgenome.resize(genome.size()/nclass);
	pstring.resize(nclass);
	double *xtemp=new double[dimension];
	FILE *fp=fopen(filename,"r");
	int d,c;
	fscanf(fp,"%d %d",&d,&c);
	value = 1.0;
	double verror;
	for(int n=0;n<nclass;n++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[n*genome.size()/nclass+j];
		int redo=0;
		pstring[n]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) 
		{
			delete[] xtemp;
			return -1e+8;
		}
		program->Parse(pstring[n]);
		for(int i=0;i<c;i++)
		{
			for(int j=0;j<dimension;j++) 
				fscanf(fp,"%lf",&xtemp[j]);
			double ytemp;
			fscanf(fp,"%lf",&ytemp);
			double sigmatemp;
			fscanf(fp, "%lf", &sigmatemp);
			double v=program->Eval(xtemp);
			if(program->EvalError() || isnan(v) || isinf(v))
			{
				delete[] xtemp;
				return -1e+8;
			}
			gdfy[i][n] = v;
			tempy[i][n]=ytemp;
			tempsigma[i][n] = sigmatemp;
		}
	}
	verror = 1.0e100;
	switch(error_function)
	{
		case MSE_ERROR:
			verror= mse_function(nclass,testcount,tempy,tempsigma, gdfy);
			break;
		case ABS_ERROR:
			verror= abs_function(nclass,testcount,tempy, tempsigma, gdfy);
			break;
		case MAX_ERROR:
			verror= max_function(nclass,testcount,tempy, tempsigma, gdfy);
			break;
		case USER_ERROR:
			verror= user_function(nclass,testcount,tempy, tempsigma, gdfy);
			break;
		case LOGLIKE_ERROR:
			verror = logLikelihood(nclass, testcount, tempy, tempsigma, gdfy);
			break;
		default:
			verror = mse_function(nclass, testcount, tempy, tempsigma, gdfy);
			break;
	}
	value=verror;
	delete[] xtemp;
	fclose(fp);
	return value;
}

double	GdfProgram::getClassError(vector<int> &genome) 
{
	vector<int> pgenome;
	vector<string> pstring;
	double value=0.0;
	pgenome.resize(genome.size()/nclass);
	pstring.resize(nclass);
	double *xtemp=new double[dimension];
	double verror=0.0;
	value=0.0;
	vector<double> classvalue;
	for(int i=0;i<testcount;i++)
	{
		double d=test_ypoint[i];
		int ifound=0;
		for(int j=0;j<classvalue.size();j++)
		{
			if(fabs(d-classvalue[j])<1e-5) 
			{
				ifound=1;
				break;
			}
		}
		if(!ifound)
		{
			int s=classvalue.size();
			classvalue.resize(s+1);
			classvalue[s]=d;
		}
	}

	for(int n=0;n<nclass;n++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[n*genome.size()/nclass+j];
		int redo=0;
		pstring[n]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) 
		{
			delete[] xtemp;
			return -1e+8;
		}
		program->Parse(pstring[n]);
		for(int i=0;i<testcount;i++)
		{
			for(int j=0;j<dimension;j++) 
			{
				xtemp[j]=test_xpoint[i][j];
			}
			double v=program->Eval(xtemp);
			if(program->EvalError() || isnan(v) || isinf(v))
			{
				delete[] xtemp;
				return -1e+8;
			}
			gdfy[i][n] = v;
			tempy[i][n]=test_ypoint[i*nclass+n];
			tempsigma[i][n] = test_sigma[i * nclass + n];
			double minValue=1e+100;
			int    minIndex=-1;
			for(int j=0;j<classvalue.size();j++)
			{
				if(fabs(classvalue[j]-v)<minValue)
				{
					minValue=fabs(classvalue[j]-v);
					minIndex=j;
				}
			}
			value=value+(fabs(test_ypoint[i]-classvalue[minIndex])>1e-5);
		}
	}
	delete[] xtemp;
	return value/testcount;
}

/* Return the test error for a particular chromosome (using the test data) */
double	GdfProgram::getTestError(vector<int> &genome) 
{
	vector<int> pgenome;
	vector<string> pstring;
	double value=0.0;
	pgenome.resize(genome.size()/nclass);
	pstring.resize(nclass);
	double *xtemp=new double[dimension];
	double verror=0.0;
	value=1.0;
	FILE *fpout=NULL;
	if(strlen(output_file))
		fpout=fopen(output_file,"w");
	for(int n=0;n<nclass;n++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[n*genome.size()/nclass+j];
		int redo=0;
		pstring[n]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) 
		{
			delete[] xtemp;
			return -1e+8;
		}
		program->Parse(pstring[n]);
		for(int i=0;i<testcount;i++)
		{
			for(int j=0;j<dimension;j++) 
			{
				xtemp[j]=test_xpoint[i][j];
			}
			double v=program->Eval(xtemp);
			if(program->EvalError() || isnan(v) || isinf(v))
			{
				delete[] xtemp;
				return -1e+8;
			}
			gdfy[i][n] = v;
			tempy[i][n]=test_ypoint[i*nclass+n];
			tempsigma[i][n] = test_sigma[i * nclass + n];
		}
	}
	if(fpout)
	{
		fprintf(fpout,"#%d\n#%d\n",nclass,testcount);
		for(int i=0;i<testcount;i++)
		{
			for(int j=0;j<nclass;j++)
			{
				fprintf(fpout,"%lf %lf ",tempy[i][j], tempsigma[i][j]);
			}
			for(int j=0;j<nclass;j++)
			{
				fprintf(fpout,"%lf ",gdfy[i][j]);
			}
			fprintf(fpout,"\n");
		}
	}
	verror = 1.0e100;
	switch (error_function)
	{
	case MSE_ERROR:
		verror = mse_function(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	case ABS_ERROR:
		verror = abs_function(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	case MAX_ERROR:
		verror = max_function(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	case USER_ERROR:
		verror = user_function(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	case LOGLIKE_ERROR:
		verror = logLikelihood(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	default:
		verror = mse_function(nclass, testcount, tempy, tempsigma, gdfy);
		break;
	}
	value=verror;
	if(fpout) fclose(fpout);
	delete[] xtemp;
	return value;
}

/* Return the train size */
int	GdfProgram::getTrainSize() const
{
	return train_ypoint.size();
}

/* Return the test size */
int	GdfProgram::getTestSize() const
{
	return test_ypoint.size();
}

static int countx(string s)
{
	int d=0;
	for(int i=0;i<s.size()-1;i++)
		if(s[i]=='x' && s[i+1]!='p') d++;
	return d;
}

/* check division zero and overflow return 1 if there is singularity, otherwise returns 0 */
int GdfProgram::checkSingularity()
{
	double* xtemp = new double[dimension]; /* dimension is dimesnion of parameter space */
	double lower_limit = check_lower;
	double upper_limit = check_upper;
	double bound_upper = upper_value;
	double bound_lower = lower_value;
	if (boundary_type == NO_BOUNDARY) return 0; /* no boundary check for divergence */
	else if (boundary_type == GLOBAL_BOUNDARY)
	{
		double check_x = lower_limit;
		while (check_x < upper_limit)
		{
			for (int j = 0; j < dimension; j++) xtemp[j] = check_x;
			double v = program->Eval(xtemp); /*실제 수식 값 계산 */
			int errorCode = program->EvalError();
			if (errorCode || v > bound_upper || v < bound_lower || isnan(v) || isinf(v))
			{
				//if(errorCode)
				//	printf("[DEBUG-hwlee]Evaluation error occurred. Code = %d\n", errorCode);
				delete[] xtemp;
				return 1;
			}
			check_x += check_delta;
		}
		delete[] xtemp;
		return 0;
	}
	/* check for local boundary */
	for (int i = 0; i < traincount-1; i++)
	{
		lower_limit = train_xpoint[i][0]; /* currently assume only on dimension */
		upper_limit = train_xpoint[i+1][0];
		bound_lower = train_ypoint[i] - lower_sigma * train_sigma[i]; /* ignore class */
		double tmp = train_ypoint[i+1] - lower_sigma * train_sigma[i+1]; /* ignore class */
		bound_lower = (bound_lower < tmp) ? bound_lower : tmp;
		bound_upper = train_ypoint[i] + upper_sigma * train_sigma[i]; /* ignore class */
		tmp = train_ypoint[i + 1] + upper_sigma * train_sigma[i + 1]; /* ignore class */
		bound_upper = (bound_upper < tmp) ? tmp: bound_upper;

		double check_x = lower_limit;
		while (check_x < upper_limit)
		{
			for (int j = 0; j < dimension; j++) xtemp[j] = check_x;
			double v = program->Eval(xtemp); /*실제 수식 값 계산 */
			int errorCode = program->EvalError();
			if (errorCode || v > bound_upper || v < bound_lower || isnan(v) || isinf(v))
			{
				//if(errorCode)
				//	printf("[DEBUG-hwlee]Evaluation error occurred. Code = %d\n", errorCode);
				delete[] xtemp;
				return 1;
			}
			check_x += check_delta;
		}
	}
	delete[] xtemp;
	return 0;

}

/* Return the fitness */
double	GdfProgram::fitness(vector<int> &genome)
{
	vector<int> pgenome;
	vector<string> pstring;
	double value=1.0;
	pgenome.resize(genome.size()/nclass);
	pstring.resize(nclass);
	double *xtemp=new double[dimension]; /* dimension is dimesnion of parameter space */
	double vmax=-1e+100;
	int ifind;
	for(int n=0;n<nclass;n++)
	{
		for(int j=0;j<pgenome.size();j++)
			pgenome[j]=genome[n*genome.size()/nclass+j];
		int redo=0;
		pstring[n]=printRandomProgram(pgenome,redo);
		if(redo>=REDO_MAX) 
		{
			delete[] xtemp;
			return -1e+100;
		}
		if(nclass>1)
		{
			ifind=countx(pstring[n]);
			if(ifind==0)
			{
				delete[] xtemp;
				return -1e+100;
			}
		}
		program->Parse(pstring[n]); /*genome의 수식 parsing*/
		if (checkSingularity())
		{
			delete[] xtemp;
			return -1e+100;
		}
		for(int i=0;i<traincount;i++)
		{
			for(int j=0;j<dimension;j++) xtemp[j]=train_xpoint[i][j];
			double v=program->Eval(xtemp); /*실제 수식 값 계산 */
			if(program->EvalError() || isnan(v) || isinf(v))
			{
				delete[] xtemp;
				return -1e+100;
			}
			gdfy[i][n] = v;
			tempy[i][n] = train_ypoint[i*nclass+n];
			tempsigma[i][n] = train_sigma[i * nclass + n];
		}
	}
	double verror = 1.0e100;
	switch (error_function)
	{
	case MSE_ERROR:
		verror = mse_function(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	case ABS_ERROR:
		verror = abs_function(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	case MAX_ERROR:
		verror = max_function(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	case USER_ERROR:
		verror = user_function(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	case LOGLIKE_ERROR:
		verror = logLikelihood(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	default:
		verror = mse_function(nclass, traincount, tempy, tempsigma, gdfy);
		break;
	}
	value=verror;
	delete[] xtemp;
	return -value;
}


/* Destructor */
GdfProgram::~GdfProgram()
{
	int maxcount=imax(traincount,testcount);
	for(int i=0;i<maxcount;i++)
	{
		delete[] gdfy[i];
		delete[] tempy[i];
		delete[] tempsigma[i];
	}
	delete[] gdfy;
	delete[] tempy;
	delete[] tempsigma;
	delete program;
	for(int i=0;i<traincount;i++) delete[] train_xpoint[i]; delete[] train_xpoint;
	for(int i=0;i<testcount;i++) delete[]  test_xpoint[i];
	if(testcount) delete[] test_xpoint;
}

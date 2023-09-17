#include "cprogram.h"
#include <stdio.h>
#include <math.h>
#include "udf.h"
#include <iostream>
#include <fstream>
#include <string>

static double Adf1Wrapper(const double *X)
{
	return udf1(X[0]);
}

static double Adf2Wrapper(const double *X)
{
	return udf2(X[0]);
}

static double Adf3Wrapper(const double *X)
{
	return udf3(X[0]);
}

static double Adf4Wrapper(const double* X)
{
	return udf4(X[0]);
}

static double Adf5Wrapper(const double* X)
{
	return udf5(X[0]);
}

static double Adf6Wrapper(const double* X)
{
	return udf6(X[0]);
}

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/* Cprogram constructor */
/* Input: the grammar filename, the dimension of the input data */
Cprogram::Cprogram(char *GrammarFile,int dim)
{
	parser.AddConstant("pi",M_PI);
	parser.AddFunction("udf1",Adf1Wrapper,1);
	parser.AddFunction("udf2",Adf2Wrapper,1);
	parser.AddFunction("udf3",Adf3Wrapper,1);
	parser.AddFunction("udf4", Adf1Wrapper, 1);
	parser.AddFunction("udf5", Adf2Wrapper, 1);
	parser.AddFunction("udf6", Adf3Wrapper, 1);
	dimension = dim;
	makeTerminals();
	loadGrammar(GrammarFile);
	makeNonTerminals();
}

/* Loads the grammar from a text file */
void	Cprogram::loadGrammar(char *GrammarFile)
{
	/* Reads the grammar file as text */
	if(GrammarFile!=NULL)
	{
		/* Load the contents of the grammar file onto the grammar string */
		long buf_size;
		char * buffer;
		string * gf = new string(GrammarFile);
		ifstream fin(gf->c_str(),ios::in|ios::binary|ios::ate);
		buf_size = fin.tellg();
		if (buf_size < 0) {
			printf("[ERROR]NO grammar file(%s) found, it should exist, stopping\n", GrammarFile);
			exit(-3);
		}
		fin.seekg(0,ios::beg);
		buffer = new char[buf_size];
		fin.read(buffer,buf_size);
		fin.close();
		string * grammar = new string(buffer);
		char * stoken = NULL;
		string * temp = new string();
		int rule_count = 0;
		temp->append(grammar->c_str());
		delete grammar;
		/* Tokenize the grammar (use the newline as delimiter) */
		stoken = strtok((char *)temp->c_str(),"\n\r");
		if(stoken==NULL) return;
		grammar_rules.push_back(string(stoken));
		rule_count++;

		while(1) {
			stoken = strtok(NULL,"\n\r");
			if(stoken==NULL) break;
			if(stoken[0]>0) {
				/* Push each line of the grammar file into the stack */
				/* The contents will be processed later */
				grammar_rules.push_back(string(stoken));
				rule_count++;
			}
		}

		// grammar_rules vector now contains the output
		int status=1;  // 1 -> LH : we are processing the left hand of the rule
		               // 2 -> RH : we are processing the right hand of the rule

		int LH=0; // LH denotes the type of the rule: 
		          //   0=Start symbol, 
		          //   1=expression, 
		          //   2=function,
		          //   3=operand
		int r;
		string *ar = new string[rule_count];
		/* Parse all the rules of the stack sequentially */
		for(int i=0; i<rule_count; i++) {
			ar[i]=grammar_rules[i];
			char * stoken = NULL;
			string * temp = new string();
			int rule_count = 0;
			temp->append(ar[i].c_str());
			stoken = strtok((char *)temp->c_str()," \t");
			if(stoken==NULL) return;
			grammar_rules.push_back(string(stoken));
			/* If current rule is in a compressed form (separated by |) create new rule */
			if(strcmp(stoken,"|")==0) 
			{
				r=newRule();
				status=2;
			} else {
				/* Parse the left hand of the rule */
				r=newRule();
				if(strcmp(stoken,"<S>")==0) LH=0;
				if(strcmp(stoken,"<expr>")==0) LH=1;
				if(strcmp(stoken,"<func>")==0) LH=2;
				if(strcmp(stoken,"<op>")==0) LH=3;
				rule_count++;
				status=1;
			}

			/* Parse the right hand of the rule */
			while(1) {
				/* Tokenize the right hand contents per space or tab */
				stoken = strtok(NULL," \t");
				if(stoken==NULL) break;
				if(stoken[0]>0) {
					grammar_rules.push_back(string(stoken));
					if(status==2) {
						/* Available symbols that fparser recognizes */
						if(strcmp(stoken,"(")==0) rule[r]->addSymbol(&Lpar);
						if(strcmp(stoken,")")==0) rule[r]->addSymbol(&Rpar);
						if(strcmp(stoken,"+")==0) rule[r]->addSymbol(&Plus);
						if(strcmp(stoken,"-")==0) rule[r]->addSymbol(&Minus);
						if(strcmp(stoken,"*")==0) rule[r]->addSymbol(&Mult);
						if(strcmp(stoken,"/")==0) rule[r]->addSymbol(&Div);
						if(strcmp(stoken,"^")==0) rule[r]->addSymbol(&Pow);
						if(strcmp(stoken,"sin")==0) rule[r]->addSymbol(&Sin);
						if(strcmp(stoken,"cos")==0) rule[r]->addSymbol(&Cos);
						if(strcmp(stoken,"exp")==0) rule[r]->addSymbol(&Exp);
						if(strcmp(stoken,"log")==0) rule[r]->addSymbol(&Log);
						if(strcmp(stoken,"abs")==0) rule[r]->addSymbol(&Abs);
						if(strcmp(stoken,"sqrt")==0) rule[r]->addSymbol(&Sqrt);
						if(strcmp(stoken,"tan")==0) rule[r]->addSymbol(&Tan);
						if(strcmp(stoken,"atan")==0) rule[r]->addSymbol(&Atan);
						if(strcmp(stoken,"asin")==0) rule[r]->addSymbol(&Asin);
						if(strcmp(stoken,"acos")==0) rule[r]->addSymbol(&Acos);
						if(strcmp(stoken,"int")==0) rule[r]->addSymbol(&Int);
						if(strcmp(stoken,"log10")==0) rule[r]->addSymbol(&Log10);
						if (strcmp(stoken, "tanh") == 0) rule[r]->addSymbol(&Tanh);
						/*GIANNIS*/
						if(strcmp(stoken,"udf1")==0)  rule[r]->addSymbol(&Adf1);
						if(strcmp(stoken,"udf2")==0)  rule[r]->addSymbol(&Adf2);
						if(strcmp(stoken,"udf3")==0)  rule[r]->addSymbol(&Adf3);
						if (strcmp(stoken, "udf4") == 0)  rule[r]->addSymbol(&Adf1);
						if (strcmp(stoken, "udf5") == 0)  rule[r]->addSymbol(&Adf2);
						if (strcmp(stoken, "udf6") == 0)  rule[r]->addSymbol(&Adf3);
						/*END OF GIANNIS*/

						if(strcmp(stoken,"<expr>")==0) rule[r]->addSymbol(&Expr);
						if(strcmp(stoken,"<op>")==0) rule[r]->addSymbol(&binaryop);
						if(strcmp(stoken,"<func>")==0) rule[r]->addSymbol(&function);
						if(strcmp(stoken,"<terminal>")==0) rule[r]->addSymbol(&terminal);
						if(strcmp(stoken,"<x>")==0) rule[r]->addSymbol(&XXlist);
					}
					if(strcmp(stoken,"::=")==0) status=2;
					if(strcmp(stoken,"|")==0) status=2;
					rule_count++;
				}
			}

			// --- Add the Rules ---
			if(LH==0) Start.addRule(rule[r]);
			if(LH==1) Expr.addRule(rule[r]);
			if(LH==2) function.addRule(rule[r]);
			if(LH==3) binaryop.addRule(rule[r]);

		}

		// --- Add the Terminals ---
		r = newRule();
		rule[r]->addSymbol(&Digit0);
		DigitList.addRule(rule[r]);

		r = newRule();
		rule[r]->addSymbol(&Digit0);
		rule[r]->addSymbol(&DigitList);
		DigitList.addRule(rule[r]);

		r = newRule();
		rule[r]->addSymbol(&DigitList);
		rule[r]->addSymbol(&Dot);
		rule[r]->addSymbol(&Digit1);
		terminal.addRule(rule[r]);

		/* XXlist are the elements of the input vector */
		r = newRule();
		rule[r]->addSymbol(&XXlist);
		terminal.addRule(rule[r]);
		for(int i=0; i<dimension; i++)
		{
			r=newRule();
			rule[r]->addSymbol(&XX[i]);
			XXlist.addRule(rule[r]);
		}

		for(int i=0; i<10; i++)
		{
			r = newRule();
			rule[r]->addSymbol(&Digit[i]);
			Digit0.addRule(rule[r]);
		}

		r=newRule();
		rule[r]->addSymbol(&Digit0);
		Digit1.addRule(rule[r]);

		r=newRule();
		rule[r]->addSymbol(&Digit0);
		rule[r]->addSymbol(&Digit0);
		Digit1.addRule(rule[r]);

	}
}

/* Create a new rule */
int	Cprogram::newRule()
{
	int r;
	r=rule.size();
	rule.resize(r+1);
	rule[r]=new Rule();
	return r;
}

/* Compile the terminal symbols */
void	Cprogram::makeTerminals()
{
	Plus.set("+",1);
	Minus.set("-",1);
	Mult.set("*",1);
	Div.set("/",1);
	Pow.set("^",1);
	Comma.set(",",1);
	Dot.set(".",1);
	Lpar.set("(",1);
	Rpar.set(")",1);
	Sin.set("sin",1);
	Cos.set("cos",1);
	Exp.set("exp",1);
	Log.set("log",1);
	Abs.set("abs",1);
	Sqrt.set("sqrt",1);
	Tan.set("tan",1);
	Atan.set("atan",1);
	Asin.set("asin",1);
	Acos.set("acos",1);
	Int.set("int",1);
	Log10.set("log10",1);
	Tanh.set("tanh", 1);
	XX.resize(dimension);
	vars="";
	for(int i=0;i<dimension;i++)
	{
		char str[100];
		sprintf(str,"x%d",i+1);
		XX[i].set(str,1);
		vars=vars+str;
		if(i<dimension-1) vars=vars+",";
	}
	Digit.resize(10);
	for(int i=0;i<10;i++)
	{
		char str[100];
		sprintf(str,"%d",i);
		Digit[i].set(str,1);
	}
	/*GIANNIS*/
	Adf1.set("udf1",1);
	Adf2.set("udf2",1);
	Adf3.set("udf3",1);
	Adf4.set("udf4", 1);
	Adf5.set("udf5", 1);
	Adf6.set("udf6", 1);
	/*END OF GIANNIS*/
}

/* Compile the non-terminal symbols */
void	Cprogram::makeNonTerminals()
{
	Start.set("START",0);
	DigitList.set("DIGITLIST",0);
	Digit0.set("DIGIT0",0);
	Digit1.set("DIGIT1",0);
	XXlist.set("XXLIST",0);
	Expr.set("EXPR",0);
	function.set("FUNCTION",0);
	binaryop.set("BINARYOP",0);
	terminal.set("TERMINAL",0);
}


/* Parse an expression using fparser */
int		Cprogram::Parse(string expr)
{
	return (parser.Parse(expr,vars)==-1);
}

/* If parse fails return the parser evaluation error */
int		Cprogram::EvalError()
{
	return	parser.EvalError();
}

/* Given a function and an input X, evaluate and return the result (Y) */
double	Cprogram::Eval( const double *X)
{
		return parser.Eval(X);
}

/* Return the start symbol */
Symbol	*Cprogram::getStartSymbol()
{
	return &Start;
}

Cprogram::~Cprogram()
{
	for(int i=0;i<rule.size();i++)
		delete rule[i];
}

#ifndef __CPROGRAM__H
#include "symbol.h"
#include "rule.h"
#include "fparser.hh"

/* The Cprogram class defines the BNF grammar internals */
/* The grammar elements (terminals, non-terminals, rules) are defined here */
/* The grammar is also loaded and parsed from text file */
class Cprogram
{
	protected:
		string		vars;
		FunctionParser 	parser;
		int			dimension;
		vector<Rule*> rule;
		Symbol		Start, Expr, function, binaryop, terminal,
					XXlist,DigitList, Digit0, Digit1;
		Symbol		Sin, Cos, Exp, Log, Abs, Sqrt,Atan,Asin,Acos;
		Symbol		Plus, Minus, Mult, Div, Pow;
		Symbol		Lpar, Rpar, Dot, Comma;
		Symbol		Tan, Int, Log10, Tanh;
		Symbol		Adf1,Adf2,Adf3;
		Symbol		Adf4, Adf5, Adf6;
		vector<Symbol>	Digit;
		vector<Symbol>	XX;
		vector<string>  grammar_rules;
		int			newRule();
		void			loadGrammar(char *GrammarFile);
		void			makeTerminals();
		void			makeNonTerminals();
	public:
		Cprogram(char *GrammarFile,int dim);
		int	Parse(string expr);
		double	Eval(const double *X);
		int	EvalError();
		Symbol	*getStartSymbol();
		~Cprogram();
};
# define __CPROGRAM__H
# endif

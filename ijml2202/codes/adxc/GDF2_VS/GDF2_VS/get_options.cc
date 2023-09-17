//#include <unistd.h>
#include <io.h>
#include "getopt.h"
#include "get_options.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int genome_count;
int genome_length;
int max_generations;
int nclass;
int error_function;
int boundary_type;
double selection_rate;
double mutation_rate;
char   grammar_file[1024];
char   train_file[1024];
char   test_file[1024];
char	output_file[1024];
char	result_file[1024];
int    random_seed=0;

/* getopt required short options */
const char *short_options="hp:c:l:s:m:p:g:t:r:n:k:e:o:f:b:";

extern char* optarg;

/* Print the program help */
void	print_usage()
{
	printf("\t-h	Print this help screen.\n"
		"\t-g	Specify grammar file (default grammar.txt).\n"
		"\t-p	Specify problem file (default ).\n"
		"\t-t	Specify test  file (default ).\n"
		"\t-c	Specify population size (default 500).\n"
		"\t-l	Specify genome length (default 200).\n"
		"\t-s	Specify selection rate (default 0.10).\n"
		"\t-m	Specify	mutation  rate (default 0.05).\n"
		"\t-n	Specify maximum number of generations (default 500).\n"
		"\t-k	Specify amount of desired classes (default 1).\n"
		"\t-e	Specify desired error function (1: MSE_ERROR(default), 2: ABS_ERROR, 3:MAX_ERROR, 4: USER_ERROR, 5: LOGLIKE_ERROR).\n"
		"\t\t	Accepted Values:\n"
		"\t\t	1. MSE FUNCTION\n"
		"\t\t	2. ABS FUNCTION\n"
		"\t\t	3. MAX FUNCTION\n"
		"\t\t	4. USER FUNCTION\n"
		"\t\t	5. Likelihood\n"
		"\t-o	Specify output log file (default ).\n"
		"\t-f	Specify result output file (default ).\n"
		"\t-r	Specify random seed (default 0).\n"
		"\t-b	Specify boundary range.\n"
		"\t\t	Accepted Values:\n"
		"\t\t	0. No boundary\n"
		"\t\t	1. Global boundary [30, 600]\n"
		"\t\t	2. Local boundary [-10sigma, 10sigma]\n");
}


/* Parse the command line arguments */
void	parse_cmd_line(int argc,char **argv)
{
	if(argc==1)
	{
		print_usage();
		exit(1);
	}
	int next_option;
	error_function = MSE_ERROR;
	boundary_type = NO_BOUNDARY;
	strcpy(output_file,"");
	strcpy(result_file, "result.dat");
	genome_count=500;
	genome_length=200;
	max_generations=1000;
	nclass=1;
	selection_rate=0.10;
	mutation_rate=0.05;
	strcpy(train_file,"");
	strcpy(test_file,"");
	strcpy(grammar_file,"grammar.txt");   /* default grammar file */
	do
	{
          next_option=getopt(argc,argv,short_options);
	      switch(next_option)
	      {
		      case 'c':
			      genome_count = atoi(optarg);
			      break;
		      case 'l':
			      genome_length = atoi(optarg);
			      break;
		      case 's':
			      selection_rate = atof(optarg);
			      break;
		      case 'm':
			      mutation_rate = atof(optarg);
			      break;
		      case 'p':
			      strcpy(train_file,optarg);
			      break;
		      case 'g':
			      strcpy(grammar_file,optarg);
			      break;
		      case 't':
			      strcpy(test_file,optarg);
			      break;
		      case 'r':
			      random_seed=atoi(optarg);
			      break;
		      case 'n':
			      max_generations=atoi(optarg);
			      break;
		      case 'k': 
			      nclass = atoi(optarg);
			      break;
		      case 'e':
			      error_function = atoi(optarg);
			      break;
			  case 'b':
				  boundary_type = atoi(optarg);
				  break;
			  case 'o':
			      strcpy(output_file,optarg);
			      break;
			  case 'f': // save the result into file
				  strcpy(result_file, optarg);
				  break;
			  case 'h':
			      print_usage();
			      exit(0);
			      break;
		      case -1:
			      break;
		      case '?':
			      print_usage();
			      exit(1);
			      break;
		      default:
			      print_usage();
			      exit(1);
			      break;
	      }
	}while(next_option!=-1);
	if (random_seed == 0)
	{
		int n = (int)time(NULL);
		if (n % 2 == 0) n++;
		random_seed = n;
	}
}

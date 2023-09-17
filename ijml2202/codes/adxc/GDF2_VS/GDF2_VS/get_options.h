# ifndef __GETOPTIONS__H
# define __GETOPTIONS__H

#define MSE_ERROR 1
#define ABS_ERROR 2
#define MAX_ERROR 3
#define USER_ERROR 4
#define LOGLIKE_ERROR 5

#define NO_BOUNDARY 0
#define GLOBAL_BOUNDARY 1
#define LOCAL_BOUNDARY 2

extern int 	genome_count;
extern int 	max_generations;
extern int 	genome_length;
extern double 	selection_rate;
extern double 	mutation_rate;
extern int    	random_seed;
extern int	nclass;
extern int	error_function;
extern int	boundary_type;
extern  char   	train_file[1024];
extern  char   	test_file[1024];
extern  char   	grammar_file[1024];
extern  void   	parse_cmd_line(int argc,char **argv);
extern	char	output_file[1024];

# endif

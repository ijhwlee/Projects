// GDF2_VS.cpp : 애플리케이션의 진입점을 정의합니다.
//

#include "GDF2_VS.h"

#include "program.h"
#include "population.h"
#include "multi_population.h"
#include <stdio.h>
#include <math.h>
#include <string>
#include "get_options.h"
#include "gdfprogram.h"

using namespace std;

extern char result_file[1024];

typedef	vector<double> Data;

int main(int argc, char** argv)
{
	parse_cmd_line(argc, argv);
	srand(random_seed);
	GdfProgram p(grammar_file, train_file, test_file);
	Population* pop;
	MultiPopulation* multipop;
	vector<int> genome;
	FILE* result = fopen(result_file, "w");
	fclose(result);
	if (nclass == 1)
	{
		pop = new Population(genome_count, genome_length, &p);
		pop->setSelectionRate(selection_rate);
		pop->setMutationRate(mutation_rate);
		pop->calcFitnessArray();
		pop->select();
		genome.resize(genome_length);
		double f = pop->getBestFitness();
		genome = pop->getBestGenome();
		string str;
		str = p.printProgram(genome);
		printf("[INFORMATION] Initial best one=======\n");
		printf("generation=%d\n%s fitness=%.10lg\n", 0, str.c_str(), f);
		result = fopen(result_file, "a");
		fprintf(result, "%d  %.16e %s", 0, f, str.c_str());
		fclose(result);
		for (int i = 1; i <= max_generations; i++)
		{
			pop->nextGeneration();
			f = pop->getBestFitness();
			genome = pop->getBestGenome();
			str = p.printProgram(genome);
			printf("generation=%d\n%s fitness=%.10lg\n", i, str.c_str(), f);
			result = fopen(result_file, "a");
			fprintf(result, "%d  %.16e %s", i, f, str.c_str());
			fclose(result);
			if (fabs(f) < 1e-7) break;
		}
		delete pop;
	}
	else
	{
		multipop = new MultiPopulation(genome_count,
			genome_length, nclass, &p);
		multipop->setSelectionRate(selection_rate);
		multipop->setMutationRate(mutation_rate);
		genome.resize(nclass * genome_length);
		string str;
		for (int i = 1; i <= max_generations; i++)
		{
			multipop->nextGeneration();
			double f = multipop->getBestFitness();
			genome = multipop->getBestGenome();
			str = p.printProgram(genome);
			printf("generation=%d\n%s\n", i, str.c_str());
			printf("fitness = %.10lg\n", f);
			result = fopen(result_file, "a");
			fprintf(result, "%d  %.16e %s", i, f, str.c_str());
			fclose(result);
			if (fabs(f) < 1e-7) break;
		}
		delete multipop;
	}
	if (strlen(test_file) != 0)
	{
		printf("APPLICATION TO %s\n", test_file);
		printf("TEST  ERROR=%.10lg\n", p.getTestError(genome));
	}
	printf("main finished\n");
	return 0;
}

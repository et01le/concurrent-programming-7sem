#include <omp.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#define SERIES_TERM_AMOUNT 1000000000L

int parse_thread_amount(int argc, char** argv);

int main(int argc, char** argv) {
    
    int thread_amount = parse_thread_amount(argc, argv);
    omp_set_num_threads(thread_amount);
    printf("Set %d threads\n", thread_amount);

    double pi = 0;

    double start = omp_get_wtime(); 
    #pragma omp parallel for reduction(+ : pi)
    for (long k = 0; k < SERIES_TERM_AMOUNT; k++) {
        double sign = k % 2 == 0? 1 : -1;
        double denom = 2 * k  + 1;
        pi += sign / denom;
    }
    double stop = omp_get_wtime();
    pi *= 4;
    
    printf("Computed value of pi = %.16f\n", pi);
    printf("Spent %.3f seconds \n", stop - start);

    return 0;
}

int parse_thread_amount(int argc, char** argv) {
    if (argc == 1) { // Binary file's name is the 1st argument
        return 1;
    }
    char* thread_amount_str = argv[1];
    for (int i = 0; i < strlen(thread_amount_str); i++) {
        if (!isdigit(thread_amount_str[i])) {
            return 1;
        }
    }
    int thread_amount = atoi(thread_amount_str);
    return thread_amount > 1 ? thread_amount : 1;
}

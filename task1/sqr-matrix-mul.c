#include <omp.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1000 // Number of rows/columns (only square matrices are used)
#define DEMO_MATRIX_SIZE 3

#define TRUE 1
#define FALSE 0
typedef short bool;

int parse_thread_amount(int argc, char** argv);
bool parse_demo_mode(int argc, char** argv);

double** allocate_sqr_matrix(size_t size);
void fill_sqr_matrix(double** matrix, size_t size, unsigned int seed);
void print_sqr_matrix(double** matrix, size_t size);
void free_matrix(double** matrix, size_t size);

void compute_product(double** matrix1, double** matrix2, double** result, size_t size);

int main(int argc, char** argv) {

    bool demo_mode = parse_demo_mode(argc, argv);
    if (demo_mode) {

        printf("Demo mode is enabled\n");

        double** matrix1 = allocate_sqr_matrix(DEMO_MATRIX_SIZE);
        double** matrix2 = allocate_sqr_matrix(DEMO_MATRIX_SIZE);
        double** product = allocate_sqr_matrix(DEMO_MATRIX_SIZE);
        if (matrix1 == NULL || matrix2 == NULL || product == NULL) {
            printf("FATAL: Unable to allocate memory\n");
            return 1;
        }
        fill_sqr_matrix(matrix1, DEMO_MATRIX_SIZE, time(NULL));
        fill_sqr_matrix(matrix2, DEMO_MATRIX_SIZE, time(NULL) + 1);

        compute_product(matrix1, matrix2, product, DEMO_MATRIX_SIZE);
       
        printf("1st matrix:\n");
        print_sqr_matrix(matrix1, DEMO_MATRIX_SIZE);
        printf("2nd matrix:\n");
        print_sqr_matrix(matrix2, DEMO_MATRIX_SIZE);
        printf("Product:\n");
        print_sqr_matrix(product, DEMO_MATRIX_SIZE);

        free_matrix(matrix1, DEMO_MATRIX_SIZE);
        free_matrix(matrix2, DEMO_MATRIX_SIZE);
        free_matrix(product, DEMO_MATRIX_SIZE);

        return 0;
    }

    int thread_amount = parse_thread_amount(argc, argv);
    omp_set_num_threads(thread_amount);
    printf("Set %d threads\n", thread_amount);

    double** matrix1 = allocate_sqr_matrix(MATRIX_SIZE);
    double** matrix2 = allocate_sqr_matrix(MATRIX_SIZE);
    double** product = allocate_sqr_matrix(MATRIX_SIZE);
    if (matrix1 == NULL || matrix2 == NULL || product == NULL) {
        printf("FATAL: Unable to allocate memory\n");
        return 1;
    }
    fill_sqr_matrix(matrix1, MATRIX_SIZE, time(NULL));
    fill_sqr_matrix(matrix2, MATRIX_SIZE, time(NULL) + 1);

    double start = omp_get_wtime();
    compute_product(matrix1, matrix2, product, MATRIX_SIZE);
    double stop = omp_get_wtime();
    
    printf("Spent %.3f seconds \n", stop - start);

    free_matrix(matrix1, MATRIX_SIZE);
    free_matrix(matrix2, MATRIX_SIZE);
    free_matrix(product, MATRIX_SIZE);

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

bool parse_demo_mode(int argc, char** argv) {
    if (argc == 1) { // Binary file's name is the 1st argument
        return FALSE;
    }
    char* demo_str = argv[1];
    if (strcmp(demo_str, "-d") == 0 || strcmp(demo_str, "--demo") == 0) {
        return TRUE;
    }
    return FALSE;
}

double** allocate_sqr_matrix(size_t size) {
    double** matrix = calloc(size, sizeof(double*));
    if (matrix == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < size; i++) {
        double* current_row = calloc(size, sizeof(double));
        if (current_row == NULL) {
            return NULL;
        }
        matrix[i] = current_row;
    }
    return matrix;
}

void fill_sqr_matrix(double** matrix, size_t size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < size; i++) {
        double* current_row = matrix[i];
        for (size_t j = 0; j < size; j++) {
           int sign = rand() % 2 == 0 ? 1 : -1;    // rand() returns an integer between 0 and RAND_MAX (both included)
           double value = (rand() % 1000) / 100.0; // Apparently, RAND_MAX=32767 for my system
           current_row[j] = value * sign;
        }
    }
}

void print_sqr_matrix(double** matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        double* current_row = matrix[i];
        for (size_t j = 0; j < size; j++) {
            printf("%5.2f ", current_row[j]); // 5 = 1 (sign) + 1 (integer part) + 1 (dot) + 2 (fractional part)
        }
        printf("\n");
    }
}

void free_matrix(double** matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void compute_product(double** matrix1, double** matrix2, double** result, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        double* result_row = result[i];
        double* matrix1_row = matrix1[i];
        for (size_t j = 0; j < size; j++) {
            double value = 0;
            for (size_t k = 0; k < size; k++) {
                value += matrix1_row[k] * matrix2[k][j];
            }
            result_row[j] = value;
        }
    }
}

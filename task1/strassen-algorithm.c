#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

#define MATRIX_SIZE 256 // Number of rows/columns (only square matrices are used)
#define DEMO_MATRIX_SIZE 4
#define THREAD_AMOUNT 7

#define TRUE 1
#define FALSE 0
typedef short bool;

bool parse_demo_mode(int argc, char** argv);

double* allocate_sqr_matrix(size_t size);
void fill_sqr_matrix(double* matrix, size_t size, unsigned int seed);
void fill_sqr_matrix_zeros(double* matrix, size_t size);
void print_sqr_matrix(double* matrix, size_t size);

double* add_up_matrices(size_t size, bool substract_last, int amount,  ...);
double* get_submatrix(double* matrix, int row_index, int col_index, size_t initial_size);
void insert_submatrix(double* matrix, int row_index, int col_index, size_t initial_size, double* submatrix);
void combine_submatrices(double* result, size_t size, double* upper_left, double* upper_right, double* lower_left, double* lower_right);
bool strassen_mul(double* matrix1, double* matrix2, double* result, size_t size, bool in_parallel);

int main(int argc, char** argv) {

    omp_set_num_threads(THREAD_AMOUNT);
    bool demo_mode = parse_demo_mode(argc, argv);
    size_t matrix_size = demo_mode ? DEMO_MATRIX_SIZE : MATRIX_SIZE;
    if (demo_mode) {
        printf("Demo mode is enabled\n");
    }

    double* matrix1 = allocate_sqr_matrix(matrix_size);
    double* matrix2 = allocate_sqr_matrix(matrix_size);
    double* product_seq = allocate_sqr_matrix(matrix_size);
    double* product_par = allocate_sqr_matrix(matrix_size);
    if (matrix1 == NULL || matrix2 == NULL || product_seq == NULL || product_par == NULL) {
        printf("FATAL: Unable to allocate memory\n");
        return 1;
    }
    fill_sqr_matrix(matrix1, matrix_size, time(NULL));
    fill_sqr_matrix(matrix2, matrix_size, time(NULL) + 1);

    double start_seq = omp_get_wtime();
    bool ok_seq = strassen_mul(matrix1, matrix2, product_seq, matrix_size, FALSE);
    double stop_seq = omp_get_wtime();

    double start_par = omp_get_wtime();
    bool ok_par = strassen_mul(matrix1, matrix2, product_par, matrix_size, TRUE);
    double stop_par = omp_get_wtime();

    if (!ok_seq || !ok_par) {
        printf("FATAL: Unable to allocate memory\n");
        return 1;
    }

    if (demo_mode) {
        printf("1st matrix:\n");
        print_sqr_matrix(matrix1, matrix_size);
        printf("2nd matrix:\n");
        print_sqr_matrix(matrix2, matrix_size);
        printf("Sequential product:\n");
        print_sqr_matrix(product_seq, matrix_size);
        printf("Parallel product:\n");
        print_sqr_matrix(product_par, matrix_size);
    } else {
        printf("Spent %.3f seconds for sequential execution\n", stop_seq - start_seq);
        printf("Spent %.3f seconds for parallel execution (%d threads)\n", stop_par - start_par, THREAD_AMOUNT);
    }

    free(matrix1);
    free(matrix2);
    free(product_seq);
    free(product_par);

    return 0;
}

bool parse_demo_mode(int argc, char** argv) {
    if (argc == 1) { // Binary file's name is the 1st argument
        return FALSE;
    }
    char* mode_str = argv[1];
    if (strcmp(mode_str, "-d") == 0 || strcmp(mode_str, "--demo") == 0) {
        return TRUE;
    }
    return FALSE;
}

double* allocate_sqr_matrix(size_t size) {
    double* matrix = calloc(size * size, sizeof(double));
    if (matrix == NULL) {
        return NULL;
    }
    return matrix;
}

void fill_sqr_matrix(double* matrix, size_t size, unsigned int seed) {
    size_t total_size = size * size;
    srand(seed);
    for (size_t i = 0; i < total_size; i++) {
        int sign = rand() % 2 == 0 ? 1 : -1;    // rand() returns an integer between 0 and RAND_MAX (both included)
        double value = (rand() % 1000) / 100.0; // Apparently, RAND_MAX=32767 for my system
        matrix[i] = value * sign;
    }
}

void fill_sqr_matrix_zeros(double* matrix, size_t size) {
    size_t total_size = size * size;
    for (size_t i = 0; i < total_size; i++) {
        matrix[i] = 0;
    }
}

void print_sqr_matrix(double* matrix, size_t size) {
    for (size_t i = 0; i < size; i++) {
        size_t row_offset = i * size;
        for (size_t j = 0; j < size; j++) {
            printf("%5.2f ", matrix[row_offset + j]); // 5 = 1 (sign) + 1 (integer part) + 1 (dot) + 2 (fractional part)
        }
        printf("\n");
    }
}

double* add_up_matrices(size_t size, bool substract_last, int amount,  ...) {
    // substract_last defines whether the last term will be added or substructed,
    // which is quite helpful option if you see how Strassen's algorithm works

    double* result = allocate_sqr_matrix(size);
    if (result == NULL) {
        return NULL;
    }
    fill_sqr_matrix_zeros(result, size);

    va_list matrices;
    va_start(matrices, amount);
    int addition_amount = substract_last ? amount - 1 : amount;
    size_t total_size = size * size;
    for (int k = 0; k < addition_amount; k++) {
        double* matrix = va_arg(matrices, double*);
        for (size_t i = 0; i < total_size; i++) {
            result[i] += matrix[i];
        }
    }
    if (substract_last) {
        double* matrix = va_arg(matrices, double*);
        for (size_t i = 0; i < total_size; i++) {
            result[i] -= matrix[i];
        }
    }
    va_end(matrices);

    return result;
}

double* get_submatrix(double* matrix, int row_index, int col_index, size_t initial_size) {
    size_t submatrix_size = initial_size / 2;
    double* submatrix = allocate_sqr_matrix(submatrix_size);
    if (submatrix == NULL) {
        return NULL;
    }
    size_t initial_row_offset = row_index * submatrix_size * initial_size;
    size_t initial_col_offset = col_index * submatrix_size;
    size_t initial_offset = initial_row_offset + initial_col_offset;
    for (size_t i = 0; i < submatrix_size; i++) {
        size_t row_offset = i * submatrix_size;
        size_t foreign_offset = initial_offset + row_offset * 2;
        for (size_t j = 0; j < submatrix_size; j++) {
            submatrix[row_offset + j] = matrix[foreign_offset + j];
        }
    }
    return submatrix;
}

void insert_submatrix(double* matrix, int row_index, int col_index, size_t initial_size, double* submatrix) {
    size_t submatrix_size = initial_size / 2;
    size_t initial_row_offset = row_index * submatrix_size * initial_size;
    size_t initial_col_offset = col_index * submatrix_size;
    size_t initial_offset = initial_row_offset + initial_col_offset;
    for (size_t i = 0; i < submatrix_size; i++) {
        size_t row_offset = i * submatrix_size;
        size_t foreign_offset = initial_offset + row_offset * 2;
        for (size_t j = 0; j < submatrix_size; j++) {
            matrix[foreign_offset + j] = submatrix[row_offset + j];
        }
    }
}

void combine_submatrices(double* result, size_t size, double* upper_left, double* upper_right, double* lower_left, double* lower_right) {
    insert_submatrix(result, 0, 0, size, upper_left);
    insert_submatrix(result, 0, 1, size, upper_right);
    insert_submatrix(result, 1, 0, size, lower_left);
    insert_submatrix(result, 1, 1, size, lower_right);
}

bool strassen_mul(double* matrix_a, double* matrix_b, double* result, size_t size, bool in_parallel) { // According to Wikipedia

    if (size == 1) {
        result[0] = matrix_a[0] * matrix_b[0];
        return TRUE;
    }

    double* a11 = get_submatrix(matrix_a, 0, 0, size);
    double* a12 = get_submatrix(matrix_a, 0, 1, size);
    double* a21 = get_submatrix(matrix_a, 1, 0, size);
    double* a22 = get_submatrix(matrix_a, 1, 1, size);
    double* b11 = get_submatrix(matrix_b, 0, 0, size);
    double* b12 = get_submatrix(matrix_b, 0, 1, size);
    double* b21 = get_submatrix(matrix_b, 1, 0, size);
    double* b22 = get_submatrix(matrix_b, 1, 1, size);

    size_t submatrix_size = size / 2;

    double* d_left   = add_up_matrices(submatrix_size, FALSE, 2, a11, a22);
    double* d_right  = add_up_matrices(submatrix_size, FALSE, 2, b11, b22);
    double* d1_left  = add_up_matrices(submatrix_size, TRUE , 2, a12, a22);
    double* d1_right = add_up_matrices(submatrix_size, FALSE, 2, b21, b22);
    double* d2_left  = add_up_matrices(submatrix_size, TRUE , 2, a21, a11);
    double* d2_right = add_up_matrices(submatrix_size, FALSE, 2, b11, b12);
    double* h1_left  = add_up_matrices(submatrix_size, FALSE, 2, a11, a12);
    double* h2_left  = add_up_matrices(submatrix_size, FALSE, 2, a21, a22);
    double* v1_right = add_up_matrices(submatrix_size, TRUE , 2, b21, b11);
    double* v2_right = add_up_matrices(submatrix_size, TRUE , 2, b12, b22);

    double* d  = allocate_sqr_matrix(submatrix_size);
    double* d1 = allocate_sqr_matrix(submatrix_size);
    double* d2 = allocate_sqr_matrix(submatrix_size);
    double* h1 = allocate_sqr_matrix(submatrix_size);
    double* h2 = allocate_sqr_matrix(submatrix_size);
    double* v1 = allocate_sqr_matrix(submatrix_size);
    double* v2 = allocate_sqr_matrix(submatrix_size);

    double* allocated[] = {a11, a12, a21, a22, b11, b12, b21, b22, d_left, d_right, d1_left, d1_right, d2_left, d2_right, h1_left, h2_left, v1_right, v2_right, d, d1, d2, h1, h2, v1, v2};
    int allocated_amount = sizeof(allocated) / sizeof(double*);
    for (int i = 0; i < allocated_amount; i++) {
        if (allocated[i] == NULL) {
            return FALSE;
        }
    }

    bool ok = TRUE;
    if (in_parallel) {
        #pragma omp parallel reduction (&&: ok)
        {
            int rank = omp_get_thread_num();
            switch(rank) {
                case 0:
                    ok = strassen_mul(d_left, d_right, d, submatrix_size, FALSE);
                    break;
                case 1:
                    ok = strassen_mul(d1_left, d1_right, d1, submatrix_size, FALSE);
                    break;
                case 2:
                    ok = strassen_mul(d2_left, d2_right, d2, submatrix_size, FALSE);
                    break;
                case 3:
                    ok = strassen_mul(h1_left, b22, h1, submatrix_size, FALSE);
                    break;
                case 4:
                    ok = strassen_mul(h2_left, b11, h2, submatrix_size, FALSE);
                    break;
                case 5:
                    ok = strassen_mul(a22, v1_right, v1, submatrix_size, FALSE);
                    break;
                case 6:
                    ok = strassen_mul(a11, v2_right, v2, submatrix_size, FALSE);
                    break;
                default:
                    printf("FATAL: Reached unreachable\n");
            }
        }
    } else {
        ok = ok && strassen_mul(d_left, d_right, d, submatrix_size, FALSE);
        ok = ok && strassen_mul(d1_left, d1_right, d1, submatrix_size, FALSE);
        ok = ok && strassen_mul(d2_left, d2_right, d2, submatrix_size, FALSE);
        ok = ok && strassen_mul(h1_left, b22, h1, submatrix_size, FALSE);
        ok = ok && strassen_mul(h2_left, b11, h2, submatrix_size, FALSE);
        ok = ok && strassen_mul(a22, v1_right, v1, submatrix_size, FALSE);
        ok = ok && strassen_mul(a11, v2_right, v2, submatrix_size, FALSE);
    }
            
    if (!ok) {
        return FALSE;
    }

    double* upper_left  = add_up_matrices(submatrix_size, TRUE , 4, d, d1, v1, h1);
    double* upper_right = add_up_matrices(submatrix_size, FALSE, 2, v2, h1);
    double* lower_left  = add_up_matrices(submatrix_size, FALSE, 2, v1, h2);
    double* lower_right = add_up_matrices(submatrix_size, TRUE , 4, d, d2, v2, h2);
    if (upper_left == NULL || upper_right == NULL || lower_left == NULL || lower_right == NULL) {
        return FALSE;
    }
    combine_submatrices(result, size, upper_left, upper_right, lower_left, lower_right);

    for (int i = 0; i < allocated_amount; i++) {
        free(allocated[i]);
    }
    free(upper_left);
    free(upper_right);
    free(lower_left);
    free(lower_right);

    return TRUE;
}

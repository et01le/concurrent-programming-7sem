#include <x86intrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>  // For timing 
#include <time.h> // For random seed

#define TRUE 1
#define FALSE 0
typedef short bool;

#define VAR_SIZE 256                                         // Size of utilized vector registry variables in bits
#define ALIGNMENT_SIZE (VAR_SIZE / 8) // = 32                // In bytes
#define VECTOR_SIZE (VAR_SIZE / (sizeof(double) * 8)) // = 4 // In double values

#define MATRIX_SIZE 1024 // Of a single axis, MUST be a multiple of VECTOR_SIZE 
#define DEMO_SIZE VECTOR_SIZE   

void free_aligned_mem(void* memory);
void* allocate_aligned_mem(size_t size_in_bytes, size_t alignment);
double* allocate_double_array(size_t size);
void fill_double_array(double* matrix, size_t size, unsigned int seed);
void fill_double_array_zeros(double* matrix, size_t total_size);
void print_matrix(double* matrix, size_t size, bool transposed);
double multiply_matrices_unvectorized(double* left_matrix, double* right_matrix_transposed, double* result_matrix, size_t size);
double multiply_matrices(double* left_matrix, double* right_matrix_transposed, double* result_matrix, size_t size);
bool run_calulations(bool demo);

int main() {
    bool ok = TRUE;
    ok = ok && run_calulations(TRUE);
    ok = ok && run_calulations(FALSE);
    return ok ? 0 : 1;
}

bool run_calulations(bool demo) {
    
    size_t size = demo ? DEMO_SIZE : MATRIX_SIZE;
    size_t total_size = size * size;
    
    double* left_matrix   = allocate_double_array(total_size);
    double* right_matrix_transposed  = allocate_double_array(total_size);
    double* result_matrix = allocate_double_array(total_size);
    if (left_matrix == NULL || right_matrix_transposed == NULL || result_matrix == NULL) {
        return 1;
    }
    fill_double_array(left_matrix, total_size, time(NULL));
    fill_double_array(right_matrix_transposed, total_size, time(NULL) + 1);

    double time_unvectorized = multiply_matrices_unvectorized(left_matrix, right_matrix_transposed, result_matrix, size);
    double time = multiply_matrices(left_matrix, right_matrix_transposed, result_matrix, size);
    if (time < 0) { // Indicates error
        return FALSE;
    }

    if (demo) {

        printf("DEMO MODE for square matrices of size = %d\n", size);
        printf("--- Left matrix ---\n");
        print_matrix(left_matrix, size, FALSE);
        printf("--- Right matrix ---\n");
        print_matrix(right_matrix_transposed, size, TRUE);
        printf("--- Vectorized result ---\n");
        print_matrix(result_matrix, size, FALSE);

        multiply_matrices_unvectorized(left_matrix, right_matrix_transposed, result_matrix, size); // Given small DEMO_SIZE, it's ok to recalculate
        
        printf("--- Unvectorized result ---\n");
        print_matrix(result_matrix, size, FALSE);

    } else {

        printf("TIMED MODE for square matrices of size = %d\n", size);
        printf("Unvectorized calculations took %.3f seconds\n", time_unvectorized);
        printf("Vectorized calculations took %.3f seconds\n", time);

    }
    
    free_aligned_mem(left_matrix);
    free_aligned_mem(right_matrix_transposed);
    free_aligned_mem(result_matrix);
    
    return TRUE;
} 

// Faced some troubles using the built-in functions, so implemented them myself
// My implementation obviously has some memory redundancy
void* allocate_aligned_mem(size_t size_in_bytes, size_t alignment) {
    void* init_memory = malloc(size_in_bytes + sizeof(void*) + alignment - 1);
    if (init_memory == NULL) {
        return NULL;
    }
    size_t offset = alignment - ((sizeof(void*) + ((size_t) init_memory)) % alignment);
    if (offset == alignment) {
        offset = 0;
    }
    offset += sizeof(void*);
    void* memory = init_memory + offset;
    *(void**) (memory - sizeof(void*)) = init_memory;
    return memory;
} 

void free_aligned_mem(void* memory) {
    void* init_memory = *(void**) (memory - sizeof(void*));
    free(init_memory);
}

double* allocate_double_array(size_t total_size) {
    double* matrix = (double*) allocate_aligned_mem(total_size * sizeof(double), ALIGNMENT_SIZE);
    if (matrix == NULL) {
        printf("FATAL: Unable to allocate memory\n");
        return NULL;
    }
    return matrix;
}

void fill_double_array(double* matrix, size_t total_size, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < total_size; i++) {
        int sign = rand() % 2 == 0 ? 1 : -1;    // rand() returns an integer between 0 and RAND_MAX (both included)
        double value = (rand() % 1000) / 100.0; // Apparently, RAND_MAX=32767 for my system
        matrix[i] = value * sign;
    }
}

void fill_double_array_zeros(double* matrix, size_t total_size) {
    for (size_t i = 0; i < total_size; i++) {
        matrix[i] = 0;
    }
}

void print_matrix(double* matrix, size_t size, bool transposed) {
    if (!transposed) {
        for (size_t i = 0; i < size; i++) {
            double* row_offset = matrix + i * size;
            for (size_t j = 0; j < size; j++) {
                printf("%5.2f ", row_offset[j]); // 5 = 1 (sign) + 1 (integer part) + 1 (dot) + 2 (fractional part)
            }
            printf("\n");
        }
    } else {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                double* row_offset = matrix + j * size;
                printf("%5.2f ", row_offset[i]); 
            }
            printf("\n");
        }
    }
}

double multiply_matrices_unvectorized(double* left_matrix, double* right_matrix_transposed, double* result_matrix, size_t size) {
    double start = omp_get_wtime();
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            size_t row_offset = i * size;
            size_t column_offset = j * size;
            double result_element = 0;
            for (size_t k = 0; k < size; k++) {
                result_element += left_matrix[row_offset + k] * right_matrix_transposed[column_offset + k];
            }
        }
    }
    double stop = omp_get_wtime();
    return stop - start;
}

double multiply_matrices(double* left_matrix, double* right_matrix_transposed, double* result_matrix, size_t size) {

    size_t mul_iter_amount = size / VECTOR_SIZE ;
    double* temp_vector = allocate_double_array(VECTOR_SIZE);
    if (temp_vector == NULL) {
        return -1;
    }

    double start = omp_get_wtime();
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {

            fill_double_array_zeros(temp_vector, VECTOR_SIZE);

            size_t row_offset    = i * size;
            size_t column_offset = j * size;

            for (size_t k = 0; k < mul_iter_amount; k++) {
                __m256d temp_prev = _mm256_loadu_pd(temp_vector);
                __m256d row       = _mm256_loadu_pd(&left_matrix[row_offset + k * VECTOR_SIZE]);
                __m256d column    = _mm256_loadu_pd(&right_matrix_transposed[column_offset + k * VECTOR_SIZE]);
                __m256d temp      = _mm256_fmadd_pd(row, column, temp_prev);
                _mm256_storeu_pd(temp_vector, temp);
            }

            double result_element = 0;
            for (size_t k = 0; k < VECTOR_SIZE; k++) {
                result_element += temp_vector[k];
            }
            result_matrix[i * size + j] = result_element;
        }
    }
    double stop = omp_get_wtime();

    free_aligned_mem(temp_vector);

    return stop - start;
}

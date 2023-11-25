#include <stdio.h>
#include <stdlib.h>
#include <time.h> // For random seed
#include <omp.h>  // For timing
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>

#define FALSE 0
#define TRUE 1
typedef short bool;

#define APPROXIMATION_ORDER 16
double APPROXIMATION_FACTOR;
double APPROXIMATION_ANGLES[APPROXIMATION_ORDER];
double TANGENT_VALUES[APPROXIMATION_ORDER];

struct vector {
    double x;
    double y;
};

#define SAMPLE_SIZE 4096 // PAY ATTENTION that this whole thing works only if SAMPLE_SIZE is a power of 2
#define MAX_TRANSFORM_DEPTH 12 // Log base 2 of SAMPLE_SIZE

void generate_approximation_values();
struct vector get_unit_vector_1st_quadrant(double angle);
struct vector get_unit_vector(double angle);
void test_cordic_algorithm();

struct vector complex_sum(struct vector a, struct vector b, bool substract);
struct vector complex_product(struct vector a, struct vector b);

struct vector* intermediate_transform(struct vector* sample, struct vector* spectrum, size_t sample_size, int depth);
struct vector* fourier_transform(double* sample, size_t sample_size, int max_depth);

void* allocate_array(size_t size);
void fill_sample_rectangle(double* sample, size_t size);
struct vector* convert_real_to_complex(double* values, size_t size);
double* convert_to_abs(struct vector* values, size_t size);
bool write_to_file(double* array, size_t size, const char* file_name);
void print_array(double* array, size_t size);
void print_complex_array(struct vector* array, size_t size, bool normalize);

// In order to comprehend my iplementation, please refer to Wikipedia pages for Cooley–Tukey FFT algorithm & CORDIC algorithm,
// as those have been the main source of my inspiration 
int main() {

    generate_approximation_values();
    test_cordic_algorithm();
 
    double* sample = (double*) allocate_array(SAMPLE_SIZE * sizeof(double));
    if (sample == NULL) {
        return 1;
    }
    fill_sample_rectangle(sample, SAMPLE_SIZE);

    struct vector* spectrum = fourier_transform(sample, SAMPLE_SIZE, MAX_TRANSFORM_DEPTH);
    if (spectrum == NULL) {
        free(sample);
        return 1;
    }

    double* spectrum_abs = convert_to_abs(spectrum, SAMPLE_SIZE);
    if (spectrum_abs == NULL) {
        free(sample);
        free(spectrum);
        return 1;
    } else {
        bool ok = write_to_file(spectrum_abs, SAMPLE_SIZE, "spectrum.txt");
        if (!ok) {
            return 1;
        }
    }

    free(sample);
    free(spectrum);
    return 0;
}

// Computation of complex exponent (or rather trigonometric functions sin & cos) can be regarded as
// the bottleneck of dicrete Fourier transform (DFT), thus becoming the main target for vectorization.
// Specifically, we will try to vectorize conventional CORDIC alogrithm for sin & cos computation.

// CORDIC algorithm relies on approximating the unit vector directed along the given angle,
// the x & y coordinates of the resulting vector will respectively be cos & sin of the angle.
// The vector is approximated by rotating the initial unit vector of [1, 0]^T
// by specific decrementing approximation angles in the direction of the given angle.
// For example, APPROXIMATION_ORDER of 10 produces approximation angles up to ~ 0.1 degrees,
// which we can be considered the error of approximation.
// Unnormalized rotation matrices are used to approximate the vector (due to efficiency reasons),
// so the result must be multiplied by a certain factor.
void generate_approximation_values() {
    APPROXIMATION_FACTOR = 1;
    for (int k = 0; k < APPROXIMATION_ORDER; k++) {
        double tan = pow(2, -k);
        APPROXIMATION_FACTOR *= sqrt(1 / (1 + tan * tan));
        APPROXIMATION_ANGLES[k] = atan(tan); // Each approximation angle is arctan(2^(-k)), where k = 0, 1, ...
        TANGENT_VALUES[k] = tan;
    }
}

// The angle parameter is considered to be from [0, pi/2]
struct vector get_unit_vector_1st_quadrant(double angle) {
        
    // The following variable determines in which direction the initial unit vector of [1, 0]^T
    // must be rotated on the current step in order to approximate the vector that we need,
    // 1 for conterclockwise, -1 for clockwise
    int rotation_direction = 1; // First rotation is conterclockwise, since we want to stay in [0, pi/2]
    double current_angle = APPROXIMATION_ANGLES[0]; // pi/4

    double elem11, elem12, elem21, elem22; 
    // Initializing rotation matrix (first rotation by pi/4 = APPROXIMATION_ANGLES[0])
    double elem11_prev =  1;
    double elem12_prev = -1; // -rotation_direction * TANGENT_VALUES[0] = -1
    double elem21_prev =  1;
    double elem22_prev =  1; //  rotation_direction * TANGENT_VALUES[0] =  1
    
    for (int i = 1; i < APPROXIMATION_ORDER; i++) {

        // Determining direction of next rotation
        if (angle >= current_angle) {
            rotation_direction = 1;
            current_angle += APPROXIMATION_ANGLES[i];
        } else {
            rotation_direction = -1;
            current_angle -= APPROXIMATION_ANGLES[i];
        }

        // TO VECTORIZE
        // Multiplying rotation matrices
        double rotation_factor = rotation_direction * TANGENT_VALUES[i];
        elem11 = elem11_prev - rotation_factor * elem21_prev;
        elem12 = elem12_prev - rotation_factor * elem22_prev;
        elem21 = rotation_factor * elem11_prev + elem21_prev;
        elem22 = rotation_factor * elem12_prev + elem22_prev;
        elem11_prev = elem11;
        elem12_prev = elem12;
        elem21_prev = elem21;
        elem22_prev = elem22;
    }

    // matrix * [1, 0]^T = 1st column of matrix + multiplying by the factor
    struct vector result = {APPROXIMATION_FACTOR * elem11, APPROXIMATION_FACTOR * elem21}; 
    return result;
}

struct vector get_unit_vector(double angle) {

    int extra_quadrants = (int) (angle * M_2_PI); // Difference between actual quadrant number (extra 2*pi loops are accounted) and 1st quadrant, M_2_PI = 2/pi
    double angle_1st_quadrant = angle - (extra_quadrants * M_PI_2); // M_PI_2 = pi/2
    int actual_quadrant = extra_quadrants % 4; // 0, 1, 2 or 3 for 1st, 2nd, 3rd and 4th quadrants (where the given angle belongs) respectively
    if (angle < 0) { // In this case the current value of actual_quadrant is either 0, -1, -2 or -3 for 4th, 3rd, 2nd and 1st quadrants respectively
        actual_quadrant = 3 + actual_quadrant;
        angle_1st_quadrant += M_PI_2;
    }
    
    struct vector unit_vector = get_unit_vector_1st_quadrant(angle_1st_quadrant);

    switch (actual_quadrant) {
        default:
            printf("FATAL: Supposed to be unreachable\n");
        case 0: {  // 1st quadrant
            return unit_vector;
        }
        case 1: { // 2nd quadrant 
            double temp = unit_vector.x;
            unit_vector.x = -unit_vector.y;
            unit_vector.y = temp;
            return unit_vector;
        }
        case 2: {  // 3rd quarant
            unit_vector.x = -unit_vector.x;
            unit_vector.y = -unit_vector.y;
            return unit_vector;
        }
        case 3: { // 4th quadrant
            double temp = unit_vector.x;
            unit_vector.x = unit_vector.y;
            unit_vector.y = - temp;
            return unit_vector;
        }
    }
}

void test_cordic_algorithm() {
    
    int max_angle_deg = 1000;
    int step = 10;
    double treshold = 0.000001;
    double convert_factor = 2 * M_PI / 360;

    printf("Running CORDIC algorithm test for %d angle values: ", 2 * max_angle_deg / step);
    printf("start = %d, stop = %d, step = %d (all in degrees)\n", -max_angle_deg, max_angle_deg, step);
    printf("Treshold error is %.6f\n", treshold);

    for (int angle_deg = -max_angle_deg; angle_deg <= max_angle_deg; angle_deg += step) {
        double angle_rad = angle_deg * convert_factor;
        struct vector unit_vector = get_unit_vector(angle_rad);
        double cos_diff = abs(unit_vector.x - cos(angle_rad));
        double sin_diff = abs(unit_vector.y - sin(angle_rad));
        if (cos_diff > treshold) {
            printf("FAIL: For angle = %d deg, cos error = %.7f", cos_diff);
        }
        if (sin_diff > treshold) {
            printf("FAIL: For angle = %d deg, sin error = %.7f", sin_diff);
        }
    }

    printf("Test run finised\n");
}

// TO VECTORIZE?
struct vector complex_sum(struct vector a, struct vector b, bool substract) {
    if (!substract) {
        struct vector sum = {a.x + b.x, a.y + b.y};
        return sum;
    } else {
        struct vector diff = {a.x - b.x, a.y - b.y};
        return diff;
    }
    
}

// TO VECTORIZE?
struct vector complex_product(struct vector a, struct vector b) { 
    struct vector product = {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    return product;
}

void* allocate_array(size_t size) {
    void* array = malloc(size);
    if (array == NULL) {
        printf("FATAL: Unable to allocate memory\n");
        return NULL;
    }
    return array;
}

void fill_sample_rectangle(double* sample, size_t size) {
    size_t left_boundary  = (size_t) (0.25 * size);
    size_t right_boundary = (size_t) (0.75 * size);
    for (size_t i = 0; i < left_boundary; i++) {
        sample[i] = 0;
    }
    for (size_t i = left_boundary; i < right_boundary; i++) {
        sample[i] = 1;
    }
    for (size_t i = right_boundary; i < size; i++) {
        sample[i] = 0;
    }
}

struct vector* convert_real_to_complex(double* values, size_t size) {
    struct vector* converted = (struct vector*) allocate_array(size * sizeof(struct vector));
    if (converted == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < size; i++) {
        struct vector complex_value = {values[i], 0};
        converted[i] = complex_value;
    }
    return converted;
}

double* convert_to_abs(struct vector* values, size_t size) {
    double* converted = (double*) allocate_array(size * sizeof(double));
    if (converted == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < size; i++) {
        double x = values[i].x;
        double y = values[i].y;
        converted[i] = sqrt(x * x + y * y);
    }
    return converted;
}

bool write_to_file(double* array, size_t size, const char* file_name) {
    FILE* file = fopen(file_name, "w");
    if (file == NULL) {
        printf("ERROR: Unable to open file\n");
        return FALSE;
    }
    for (size_t i = 0; i < size; i++) {
        fprintf(file, "%.6f ", array[i]);
    }
    fclose(file);
    return TRUE;
}

void print_array(double* array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%.3f ", array[i]);
    }
}

void print_complex_array(struct vector* array, size_t size, bool normalize) {
    for (size_t i = 0; i < size; i++) {
        char* str = array[i].y >= 0 ? "%.3f+%.3fi " : "%.3f%.3fi ";
        printf(str, array[i].x / (normalize ? sqrt(size) : 1), array[i].y / (normalize ? sqrt(size) : 1));
        // Apparently, if you type "discrete Fourier transform of [0, 1, 1, 2, 3, 5, 8, 13, 21]" in Wolfram Alpha (which is my QA),
        // the result will be normalized by square root of the sample size (the sample provided is purely random and doesn't matter).
        // Moreover, my inplementation produces compelx conjugation of Wolrfam Alpha's version. 
    }
}

struct vector* intermediate_transform(struct vector* sample, struct vector* spectrum, size_t sample_size, int depth) {

    // The range of depth parameter passed to the function is [1, ..., log base 2 of sample_size] (thus, sample_size must be a power of 2).
    // depth = 0 can be considered the initial sample. Maximum depth represents the Fourier transform itself.

    // On each step (or rather depth) of FFT alogrithm an array containing the result of intermediate calculations is produced.
    // Every pair of values of the array shares another pair of values from the previous step that are used in their calculations. 
    // However, these values (on the current step) aren't adjacent in the array, but placed within some offset, which is crusial for the alogrithm.
    // The pair of values from the previous step, which is used in calculations, is also placed within a certain offset (of sample_size / 2).
    // A handwritten scheme that migh provide better comprehension of this magic is attached in the repo.
    size_t offset = 1 << (depth - 1); // 2 ^ (depth - 1)
    size_t sample_size_half = sample_size / 2;

    // Due to previously intorduced offsetting, the array pointer must be shifted after each subiteration.
    // In this context, each intermediate_transform() invocation is considered a higher level interation.
    size_t sub_iter_size = offset; // How many pairs of values are calculated during a subiteration
    size_t sub_iter_amount = sample_size / (2 * sub_iter_size);
    size_t shift = 2 * offset;

    struct vector* initial_spectrum_ptr = spectrum;

    size_t sample_elem_index = 0;
    for (size_t sub_iter = 0; sub_iter < sub_iter_amount; sub_iter++) {
        for (size_t spectrum_elem_index = 0; spectrum_elem_index < sub_iter_size; spectrum_elem_index++) {
            struct vector complex_factor = get_unit_vector(-2 * M_PI * spectrum_elem_index / (2 * sub_iter_size));
            spectrum[spectrum_elem_index]          = complex_sum(sample[sample_elem_index], complex_product(complex_factor, sample[sample_elem_index + sample_size_half]), FALSE);
            spectrum[spectrum_elem_index + offset] = complex_sum(sample[sample_elem_index], complex_product(complex_factor, sample[sample_elem_index + sample_size_half]), TRUE);
            sample_elem_index++;
        }
        spectrum += shift;
    }
    
    return initial_spectrum_ptr;
}

// max_depth must be log base 2 of sample_size
struct vector* fourier_transform(double* sample_raw, size_t sample_size, int max_depth) {

    struct vector* sample = convert_real_to_complex(sample_raw, sample_size);
    if (sample == NULL) {
        return NULL;
    }
    struct vector* spectrum = allocate_array(sample_size * sizeof(struct vector));
    if (spectrum == NULL) {
        free(sample);
        return NULL;
    }

    struct vector* temp_ptr;
    double start = omp_get_wtime();
    for (int depth = 1; depth <= max_depth; depth++) {
        temp_ptr = sample;
        sample = intermediate_transform(sample, spectrum, sample_size, depth);
        spectrum = temp_ptr;
    }
    double stop = omp_get_wtime();
    spectrum = sample; // Pointers are shuffled a bit during calculations
    sample = temp_ptr;

    printf("Calculated unvectorized FFT within %.3f seconds\n", stop - start);

    free(sample);
    return spectrum;
}
#ifndef MATRIX_H
#define MATRIX_H
#ifdef __cplusplus
extern "C" {
#endif
typedef struct matrix{
    int rows, cols;
    double *data;
} matrix;

void free_matrix(matrix m);
matrix make_matrix(int rows, int cols);
matrix copy_matrix(matrix m);
void print_matrix(matrix m);
#ifdef __cplusplus
}
#endif
#endif

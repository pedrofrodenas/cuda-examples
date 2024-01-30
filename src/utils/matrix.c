#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

void free_matrix(matrix m)
{
    free(m.data);
}

matrix make_empty_matrix(int rows, int cols)
{
    matrix out;
    out.data = 0;
    out.rows = rows;
    out.cols = cols;
    return out;
}

matrix make_matrix(int rows, int cols)
{
    matrix m = make_empty_matrix(rows, cols);
    m.data = calloc(rows*cols, sizeof(float));
    return m;
}

matrix copy_matrix(matrix m)
{
    int i;
    matrix c = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows*m.cols; ++i){
        c.data[i] = m.data[i];
    }
    return c;
}

void print_matrix(matrix m)
{
    printf("|  ");
    for (int i=0; i<m.rows; ++i)
    {
        printf("|  ");
        for (int j=0; j<m.cols; ++j)
        {
            printf("%15.7f ", m.data[i*m.cols + j]);
        }
        printf(" |\n");
    }
    printf("|__");
}

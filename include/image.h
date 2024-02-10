#ifndef IMAGE_H
#define IMAGE_H
#include <stdio.h>

#define TWOPI 6.2831853

#ifdef __cplusplus
extern "C" {
#endif

// DO NOT CHANGE THIS FILE

typedef struct{
    int c,h,w;
    float *data;
} image;


// Loading and saving
image make_image(int c, int h, int w);
image load_image(char *filename);
void save_image(image im, const char *name);
void save_png(image im, const char *name);
void save_image_binary(image im, const char *fname);
image load_image_binary(const char *fname);
void save_png(image im, const char *name);
void free_image(image im);

// Pixel write and read
float get_pixel(image im, int c, int h, int w);
void set_pixel(image im, int c, int h, int w, float v);

void scale_image(image im, int c, float v);

// Normalization
void l1_normalize(image im);
void feature_normalize(image im);


#ifdef __cplusplus
}
#endif
#endif


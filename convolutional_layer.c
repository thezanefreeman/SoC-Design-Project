#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <assert.h>
#include "convolutional_layer.h"
#include "box.h"
#include <math.h>
#include <signal.h>

#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
#define FIXED_POINT_FRACTIONAL_BITS 12
#define UPPER 0xFF00
#define LOWER 0x00FF
typedef uint32_t fixed_point_t;
uint32_t double_to_fixed(double input);
double fixed_to_double(uint32_t input);

static inline float logistic_activate(float x){return 1.f/(1.f + expf(-x));}


inline double fixed_to_double(uint32_t input)
{
    return ((double)input / (double)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

inline uint32_t double_to_fixed(double input)
{
    return (uint32_t)(input * (1 << FIXED_POINT_FRACTIONAL_BITS));
}



#define READ_CMD  (0x0 << 31)
#define WRITE_CMD (0x1 << 31)

volatile int det_int = 0;

// signal handler for receiving events from hardware driver
void sighandler(int signo)
{
  if(signo==SIGIO)
    {
      det_int++;
      printf("\nInterrupt detected\n");
    }
  
  return;
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    // volatile unsigned int *base, *address, *base_dma;
    // unsigned long addr1, addr2, addr3, addr4, addr0, offset, value, DMA, DMA2, mem;
    // unsigned long val, result;

    // //Predefined addresses.
    // addr0 = 0xa0000000ul;  // DEBUG_TIME
    // addr1 = 0xa0000004ul;  // DEBUG_WRITE
    // addr2 = 0xa0000008ul;  // DEBUG_STOP
    // addr3 = 0xa000000Cul;  // DEBUG_IRQ
    // addr4 = 0xa0000010ul;  // DEBUG_REALTIME
    // mem   = 0xa0800000ul;

    // //Open memory as a file
    // int fd = open("/dev/mem", O_RDWR|O_SYNC);
    // if(!fd)
    //     {
    //     printf("Unable to open /dev/mem.  Ensure it exists (major=1, minor=1)\n");
    //     return -1;
    //     }	

    // //Map the physical base address to local pointer (in virtual address space)
    // base = (unsigned int *)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, addr0 & ~MAP_MASK);	
    // base_dma = (unsigned int *)mmap(NULL, MAP_SIZE*16, PROT_READ|PROT_WRITE, MAP_SHARED, fd, mem & ~MAP_MASK);	
    // if((base == MAP_FAILED))
    // {
    //     printf("mapping failed\n");
    //     fflush(stdout);
    // }

    // double input;
    // uint32_t r;
    // double f;

    int i, j, k;


    // address = base_dma;
    // *address = M;
    // address = base_dma + 1;
    // *address = N;
    // address = base_dma + 2;
    // *address = K;
    // address = base_dma + 3;
    // *address = ALPHA;
    // address = base_dma + 4;
    // *address = lda;
    // address = base_dma + 5;
    // *address = ldb;
    // address = base_dma + 6;
    // *address = ldc;
    // address = base_dma + 7;
    // *address = 0; 
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            float A_PART = ALPHA*A[i*lda + k];
            for (j = 0; j < N; ++j) {
                C[i*ldc + j] += A_PART*B[k*ldb + j];
            }
        }
    }
    // for (i = 0; i < M; ++i) {
    //     for (k = 0; k < K; ++k) {
    //         PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
    //         r = double_to_fixed(A_PART);
    //         address = base_dma + 8;
    //         *address = r;

    //         for (j = 0; j < 5; ++j) {
    //             r = double_to_fixed(B[k*ldb + j]);
    //             //printf("B value #%d: %f\n", j, B[k*ldb + j]);
    //             address = base_dma + j + 9;
    //             *address = r;

    //             r = double_to_fixed(C[i*ldc + j]);
    //             //printf("C value #%d: %f\n", j, C[i*ldc + j]);
    //             address = base_dma + j + 5000 + 9;
    //             *address = r;
    //         }
    //         printf("Done Copying\n");

    //         address = base + 5;
    //         *address = 1;
    //         address = base_dma + 7;
    //         result = *address;
    //         while(!result){
    //             result = *address;
    //         }
    //         for(j = 0; j < N; ++j){
    //             C[i*ldc + j] += A_PART*B[k*ldb + j];
    //         }
            
    //     }
    // }










    //return 0;
}


void activate_array_cpu_custom(float *x, const int n, const ACTIVATION a)
{
    int i;
    if (a == LINEAR)
    {
    }
    else if (a == LEAKY)
    {
        for (i = 0; i < n; ++i) {
            x[i] = (x[i]>0) ? x[i] : .1*x[i];
        }
    }
    else {
        for (i = 0; i < n; ++i) {
            //x[i] = activate(x[i], a);
            x[i] = logistic_activate(x[i]);
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
        int t;
        #pragma omp parallel for
        for (t = 0; t < M; ++t) {
            if (!TA && !TB){
                gemm_nn(1, N, K, ALPHA, A + t*lda, lda, B, ldb, C + t*ldc, ldc);
            }
        }
}

inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {return (unsigned)(a) < (unsigned)(b);}
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col)
{
    const int output_h = (height + 2 * pad_h -
        (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w -
        (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    int channel, kernel_row, kernel_col, output_rows, output_col;
    for (channel = channels; channel--; data_im += channel_size) {
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for (output_rows = output_h; output_rows; output_rows--) {
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for (output_col = output_w; output_col; output_col--) {
                            *(data_col++) = 0;
                        }
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}

int convolutional_out_height(convolutional_layer l){return (l.h + 2*l.pad - l.size) / l.stride_y + 1;}
int convolutional_out_width(convolutional_layer l){return (l.w + 2*l.pad - l.size) / l.stride_x + 1;}

size_t get_workspace_size32(layer l){
    if (l.xnor) {
        size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
        size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
        if (workspace_size < re_packed_input_size) workspace_size = re_packed_input_size;
        return workspace_size;
    }
    return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
}
size_t get_convolutional_workspace_size(layer l) {
    size_t workspace_size = get_workspace_size32(l);
    size_t workspace_size16 = 0;//get_workspace_size16(l);
    if (workspace_size16 > workspace_size) workspace_size = workspace_size16;
    return workspace_size;
}

void free_convolutional_batchnorm(convolutional_layer *l)
{
    if (!l->share_layer) {
        if (l->scales)          free(l->scales),            l->scales = NULL;
        if (l->scale_updates)   free(l->scale_updates),     l->scale_updates = NULL;
        if (l->mean)            free(l->mean),              l->mean = NULL;
        if (l->variance)        free(l->variance),          l->variance = NULL;
        if (l->mean_delta)      free(l->mean_delta),        l->mean_delta = NULL;
        if (l->variance_delta)  free(l->variance_delta),    l->variance_delta = NULL;
        if (l->rolling_mean)    free(l->rolling_mean),      l->rolling_mean = NULL;
        if (l->rolling_variance) free(l->rolling_variance),  l->rolling_variance = NULL;
        if (l->x)               free(l->x),                 l->x = NULL;
        if (l->x_norm)          free(l->x_norm),            l->x_norm = NULL;
    }
}

convolutional_layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, convolutional_layer *share_layer, int assisted_excitation, int deform, int train)
{
    int total_batch = batch*steps;
    int i;
    convolutional_layer l = { (LAYER_TYPE)0 };
    l.type = CONVOLUTIONAL;
    l.train = train;

    if (xnor) groups = 1;   // disable groups for XNOR-net
    if (groups < 1) groups = 1;

    const int blur_stride_x = stride_x;
    const int blur_stride_y = stride_y;
    l.antialiasing = antialiasing;
    if (antialiasing) {
        stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
    }

    l.wait_stream_id = -1;
    l.deform = deform;
    l.assisted_excitation = assisted_excitation;
    l.share_layer = share_layer;
    l.index = index;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch = batch;
    l.steps = steps;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;
    l.learning_rate_scale = 1;
    l.nweights = (c / groups) * n * size * size;

    if (l.share_layer) {
        if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n) {
            printf(" Layer size, nweights, channels or filters don't match for the share_layer");
            getchar();
        }

        l.weights = l.share_layer->weights;
        l.weight_updates = l.share_layer->weight_updates;

        l.biases = l.share_layer->biases;
        l.bias_updates = l.share_layer->bias_updates;
    }
    else {
        l.weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.biases = (float*)xcalloc(n, sizeof(float));

        if (train) {
            l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
            l.bias_updates = (float*)xcalloc(n, sizeof(float));

            l.weights_ema = (float*)xcalloc(l.nweights, sizeof(float));
            l.biases_ema = (float*)xcalloc(n, sizeof(float));
        }
    }
    float scale = sqrt(2./(size*size*c/groups));
    if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) {      
        for (i = 0; i < l.nweights; ++i) l.weights[i] = 1;   // rand_normal();
    }
    else {
        for (i = 0; i < l.nweights; ++i){
            l.weights[i] = scale*rand_uniform(-1, 1);   // rand_normal();
        } 
    }


    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;
    l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
    //printf("Got Here DESIGN REVIEW");
    l.forward = forward_convolutional_layer;
    if(binary){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.cweights = (char*)xcalloc(l.nweights, sizeof(char));
        l.scales = (float*)xcalloc(n, sizeof(float));
    }
    if(xnor){
        l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
        l.binary_input = (float*)xcalloc(l.inputs * l.batch, sizeof(float));

        int align = 32;// 8;
        int src_align = l.out_h*l.out_w;
        l.bit_align = src_align + (align - src_align % align);

        l.mean_arr = (float*)xcalloc(l.n, sizeof(float));

        const size_t new_c = l.c / 32;
        size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
        l.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

        l.lda_align = 256;  // AVX2
        int k = l.size*l.size*l.c;
        size_t k_aligned = k + (l.lda_align - k%l.lda_align);
        size_t t_bit_input_size = k_aligned * l.bit_align / 8;
        l.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
    }

    if(batch_normalize){
        if (l.share_layer) {
            l.scales = l.share_layer->scales;
            l.scale_updates = l.share_layer->scale_updates;
            l.mean = l.share_layer->mean;
            l.variance = l.share_layer->variance;
            l.mean_delta = l.share_layer->mean_delta;
            l.variance_delta = l.share_layer->variance_delta;
            l.rolling_mean = l.share_layer->rolling_mean;
            l.rolling_variance = l.share_layer->rolling_variance;
        }
        else {
            l.scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                l.scales[i] = 1;
            }
            if (train) {
                l.scales_ema = (float*)xcalloc(n, sizeof(float));
                l.scale_updates = (float*)xcalloc(n, sizeof(float));

                l.mean = (float*)xcalloc(n, sizeof(float));
                l.variance = (float*)xcalloc(n, sizeof(float));

                l.mean_delta = (float*)xcalloc(n, sizeof(float));
                l.variance_delta = (float*)xcalloc(n, sizeof(float));
            }
            l.rolling_mean = (float*)xcalloc(n, sizeof(float));
            l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        }
    }

    if(adam){
        l.adam = 1;
        l.m = (float*)xcalloc(l.nweights, sizeof(float));
        l.v = (float*)xcalloc(l.nweights, sizeof(float));
        l.bias_m = (float*)xcalloc(n, sizeof(float));
        l.scale_m = (float*)xcalloc(n, sizeof(float));
        l.bias_v = (float*)xcalloc(n, sizeof(float));
        l.scale_v = (float*)xcalloc(n, sizeof(float));
    }
    l.workspace_size = get_convolutional_workspace_size(l);
    l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
    if (l.xnor) l.bflops = l.bflops / 32;
    if (l.xnor && l.use_bin_output) fprintf(stderr, "convXB");
    else if (l.xnor) fprintf(stderr, "convX ");
    else if (l.share_layer) fprintf(stderr, "convS ");
    else if (l.assisted_excitation) fprintf(stderr, "convAE");
    else fprintf(stderr, "conv  ");

    if (groups > 1) fprintf(stderr, "%5d/%4d ", n, groups);
    else           fprintf(stderr, "%5d      ", n);

    if (stride_x != stride_y) fprintf(stderr, "%2dx%2d/%2dx%2d ", size, size, stride_x, stride_y);
    else {
        if (dilation > 1) fprintf(stderr, "%2d x%2d/%2d(%1d)", size, size, stride_x, dilation);
        else             fprintf(stderr, "%2d x%2d/%2d   ", size, size, stride_x);
    }

    fprintf(stderr, "%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", w, h, c, l.out_w, l.out_h, l.out_c, l.bflops); 
    if (l.antialiasing) {
        printf("AA:  ");
        l.input_layer = (layer*)calloc(1, sizeof(layer));
        int blur_size = 3;
        int blur_pad = blur_size / 2;
        if (l.antialiasing == 2) {
            blur_size = 2;
            blur_pad = 0;
        }
        *(l.input_layer) = make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
        const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
        int i;
        if (blur_size == 2) {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 4.f;
                l.input_layer->weights[i + 1] = 1 / 4.f;
                l.input_layer->weights[i + 2] = 1 / 4.f;
                l.input_layer->weights[i + 3] = 1 / 4.f;
            }
        }
        else {
            for (i = 0; i < blur_nweights; i += (blur_size*blur_size)) {
                l.input_layer->weights[i + 0] = 1 / 16.f;
                l.input_layer->weights[i + 1] = 2 / 16.f;
                l.input_layer->weights[i + 2] = 1 / 16.f;

                l.input_layer->weights[i + 3] = 2 / 16.f;
                l.input_layer->weights[i + 4] = 4 / 16.f;
                l.input_layer->weights[i + 5] = 2 / 16.f;

                l.input_layer->weights[i + 6] = 1 / 16.f;
                l.input_layer->weights[i + 7] = 2 / 16.f;
                l.input_layer->weights[i + 8] = 1 / 16.f;
            }
        }
        for (i = 0; i < n; ++i) l.input_layer->biases[i] = 0;
    }

    return l;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void convolutional_zane_fill_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    if (INCX == 1 && ALPHA == 0) {memset(X, 0, N * sizeof(float));}
    else {for (i = 0; i < N; ++i) X[i*INCX] = ALPHA;}
}

int convolutional_count = 0;
void forward_convolutional_layer(convolutional_layer l, network_state state)
{
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);


    
    convolutional_zane_fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    int m = l.n / l.groups;
    int k = l.size*l.size*l.c / l.groups;
    int n = out_h*out_w;
    static int u = 0;
    u++;
    float *a = l.weights; //+j*l.nweights / l.groups;
    float *b = state.workspace;
    float *c = l.output;// +(i*l.groups + j)*n*m;
    float *im = state.input;// + (0*l.groups + 0)*(l.c / l.groups)*l.h*l.w;
    if (l.size == 1 && l.stride == 1 && l.dilation == 1) {
        b = im;
    }
    else {
        im2col_cpu_ext(im,   // input
            l.c / l.groups,     // input channels
            l.h, l.w,           // input size (h, w)
            l.size, l.size,     // kernel size (h, w)
            l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
            l.stride_y, l.stride_x, // stride (h, w)
            l.dilation, l.dilation, // dilation (h, w)
            b);                 // output

    }
    convolutional_count++;
    printf("Layer Number: %d\n", convolutional_count);
    int t;
    printf("Matrix A is %dx%d\n", m, k);
    printf("Matrix B is %dx%d\n", k, n);
    printf("Matrix C is %dx%d\n", m, n);
    //#pragma omp parallel for
    for (t = 0; t < m; ++t) {
        //gemm_nn(1, n, k, 1, a + t*k, k, b, n, c + t*n, n);
        //gemm_nn(M, N, K, ALPHA,*A,lda,*B,ldb,*C,ldc);
        int y,z;
        float *A = a + t*k;
        float *C = c + t*n;

        //printf("C matrix begin: %f\n", C[1]);
        for (z = 0; z < k; ++z) {
            float A_PART = A[z];
            for (y = 0; y < n; ++y) {
                C[y] += A_PART*b[z*n + y];
            }
        }
    }

    unsigned long volatile trig, val_A, val_B, result;
    unsigned long volatile gie, iie;
    struct sigaction action;
    int fd;

    
    //Ensure proper usage

    // install signal handler
    sigemptyset(&action.sa_mask);
    sigaddset(&action.sa_mask, SIGIO);

    action.sa_handler = sighandler;
    action.sa_flags=0;

    sigaction(SIGIO, &action, NULL);

    // open hardware device (driver)
    fd=open("/dev/fpga", O_RDWR);
    if(fd < 0)
    {
        printf("Unable to open /dev/fpga.  Ensure it exists!\n");
    }
    fcntl(fd, F_SETOWN, getpid());
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL)|O_ASYNC);

    // enable FPGA interrupts (global and IP)
    ioctl(fd, READ_CMD + 0x1, &gie);
    gie = gie | 0x00000001;
    ioctl(fd, WRITE_CMD + 0x1, &gie);

    iie = 0x1;
    ioctl(fd, WRITE_CMD + 0x2, &iie);

    // perform C += A*B;
    val_A = m;
    val_B = n;

    // write A
    ioctl(fd, WRITE_CMD + 0x4, &val_A);
    printf("A is %lu\n", val_A);
    
    // write B
    ioctl(fd, WRITE_CMD + 0x6, &val_B);
    printf("B is %lu\n", val_B);
    
    // trigger MAC operation
    trig = 0x1;
    ioctl(fd, WRITE_CMD, &trig);

    // wait for interrupt
    while(!det_int) continue;

    // read result
    ioctl(fd, READ_CMD + 0x8, &result);

    printf("C += A*B is %lu\n", result);

    //In the end, close the device driver
    close(fd);



    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);
}

void assisted_excitation_forward(convolutional_layer l, network_state state)
{
    const int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);
    float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches));

    if (l.assisted_excitation > 1) {
        if (iteration_num > l.assisted_excitation) alpha = 0;
        else alpha = (1 + cos(3.141592 * iteration_num / l.assisted_excitation));
    }
    float *a_avg = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));
    float *g = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));

    int b;
    int w, h, c;

    l.max_boxes = state.net.num_boxes;
    l.truths = l.max_boxes*(4 + 1);

    for (b = 0; b < l.batch; ++b)
    {
        // calculate G
        int t;
        for (t = 0; t < state.net.num_boxes; ++t) {
            box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
            if (!truth.x) break;  // continue;

            int left = floor((truth.x - truth.w / 2) * l.out_w);
            int right = ceil((truth.x + truth.w / 2) * l.out_w);
            int top = floor((truth.y - truth.h / 2) * l.out_h);
            int bottom = ceil((truth.y + truth.h / 2) * l.out_h);

            for (w = left; w <= right; w++) {
                for (h = top; h < bottom; h++) {
                    g[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
                }
            }
        }
    }

    for (b = 0; b < l.batch; ++b)
    {
        // calculate average A
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++) {
                    a_avg[w + l.out_w*(h + l.out_h*b)] += l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))];
                }
                a_avg[w + l.out_w*(h + l.out_h*b)] /= l.out_c;  // a_avg / d
            }
        }
    }

    // change activation
    for (b = 0; b < l.batch; ++b)
    {
        for (w = 0; w < l.out_w; w++) {
            for (h = 0; h < l.out_h; h++) {
                for (c = 0; c < l.out_c; c++)
                {
                    l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] +=
                        alpha *
                        g[w + l.out_w*(h + l.out_h*b)] *
                        a_avg[w + l.out_w*(h + l.out_h*b)];
                }
            }
        }
    }
    free(g);
    free(a_avg);
}


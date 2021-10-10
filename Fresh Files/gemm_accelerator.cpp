#include "systemc.h"
SC_MODULE (gemm_accelerator) {
    sc_signal<bool> request;//Request Signal from CPU
    sc_int<32> ram[7600000];
//------------GEMM Inputs and Outputs---------------------
    sc_in<int> M;
    sc_in<int> N;
    sc_in<int> K;
    sc_in<float> ALPHA;
    sc_in<float> *A;
    sc_in<float> *B;
    sc_in<float> *C;
    sc_in<int> lda;
    sc_in<int> ldb;
    sc_in<int> ldc;
    sc_out<float> output;
    sc_out<float> *output_pointer;
//------------Local Variables---------------------
    sc_int<32> i;
    sc_int<32> j;
    sc_int<32> k;
    sc_int<32> B_FX;
    sc_int<32> C_FX;
    sc_int<32> A_FX;
    sc_int<32> FixedPointValue;
    sc_int<32> scale;
    float B_Float;
    float A_Float;
    float C_Float;
    float A_PART_TWO;
//------------GEMM Constructor--------------------- 
    SC_CTOR (gemm_accelerator) {
        converter* a_converter = new converter("a_converter");
        converter* b_converter = new converter("b_converter");
        converter* c_converter = new converter("c_converter");
        a_converter->input(A_Float);
        a_converter->output(A_FX);
        
        b_converter->input(B_Float);
        b_converter->output(B_FX);
        
        c_converter->input(C_Float);
        c_converter->output(C_FX);
        
        SC_THREAD(gemm_nn);
        sensitive << request;
        //SC_THREAD(initialize_ram);
    }
//------------GEMM Function---------------------
    void gemm_nn() {
        while(true){
            wait();
            scale = 16;
            for (i = 0; i < M; ++i) {
                for (k = 0; k < K; ++k) {
                A_Float = A[i * lda + k] * (1<<scale);
                wait(done_converting);
                FixedPointValue = ALPHA * A_FX;
                A_PART_TWO = (float)(FixedPointValue)/(1<<scale);
                    PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
                    for (j = 0; j < N; ++j) {
                    B_Float = B[k * ldb + j] * (1<<scale);
                    C_Float = C[i * ldc + j] * (1<<scale);
                    wait(done_converting);
                    wait(done_converting);
                    FixedPointValue = A_PART_TWO*B_FX;
                    FixedPointValue = C_FX + FixedPointValue;
                        C[i*ldc + j] = (float)(FixedPointValue)/(1<<scale);
                    }
                }
            }
        }
    }
//------------Floating to Fixed SubModule---------------------
    SC_MODULE(converter){
        sc_in<float> input;
        sc_out<int> output;
        SC_CTOR(converter){
            SC_THREAD(roundup);
            sensitive << input;
        }
//------------Round Up Function---------------------
        void roundup()
        {
            while(true){
                wait();
                float fp_number = input.read();
                int	fx_number	=	(int)fp_number;
                if(fp_number-fx_number>=0.5)	fx_number++;
                output.write(fx_number);
                done_converting.notify();
            }
        }
    };
};
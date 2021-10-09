#include "systemc.h" 
 SC_MODULE (hardware_accelerator) {
    sc_in<float> Data;///This probably needs to be an SC_FIFO??
    sc_in<bool> reset; //might be good to have a reset if anything goes wrong.... idk?
    sc_out<float> output[1024]; // i don't know how big the output matrix is going to be?
//------------Internal Hardware Variables---------------------
    sc_uint<1024> ASize;
    sc_uint<1024> BSize;
    sc_uint<1024> CSize;
//------------Code Starts Here-------------------------
//This will not work as of right now
 22   void gemm() {
    int i, j, k, B_FX, C_FX, A_FX, FixedPointValue, scale;
    float B_Float, A_Float, C_Float, A_PART_TWO;
    scale = 16;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
		A_Float = A[i * lda + k] * (1<<scale);
		A_FX = (int)A_Float;
		if(A_Float - B_FX >= 0.5){
			A_FX++;
		}
		FixedPointValue = ALPHA * A_FX;
		A_PART_TWO = (float)(FixedPointValue)/(1<<scale);
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
		    B_Float = B[k * ldb + j] * (1<<scale);
		    B_FX = (int)B_Float;
		    if(B_Float - B_FX >= 0.5){
			    B_FX++;
		    }
		    FixedPointValue = A_PART_TWO*B_FX;
		    C_Float = C[i * ldc + j] * (1<<scale);
		    C_FX = (int)C_Float;
		    if(C_Float - C_FX >= 0.5){
			    C_FX++;
		    }
		    FixedPointValue = C_FX + FixedPointValue;
                C[i*ldc + j] = (float)(FixedPointValue)/(1<<scale);
            }
        }
    }
}
    SC_CTOR(hardware_accelerator) {
        cout<<"Executing New Gemm Calculation"<<endl;
       SC_METHOD(gemm);
        sensitive << reset;
        sensitive << Data;
       } 

 };
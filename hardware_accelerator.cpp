#include "systemc.h" 
SC_MODULE (hardware_accelerator) {
//------------Internal Hardware Variables---------------------
	sc_in <int> M, N, K;//size of arrays 
	sc_in <float> ALPHA;
	sc_in <float>* A, B; 
	sc_out <float>* C;
	sc_in <int> lda, ldb, ldc;
//------------Code Starts Here-------------------------
//This will not work as of right now
	void gemm() {
		int i, j, k, B_FX, C_FX, A_FX, FixedPointValue, scale;
		float B_Float, A_Float, C_Float, A_PART_TWO;
		scale = 16;
		for (i = 0; i < M; ++i) {
			for (k = 0; k < K; ++k) {
			//Convert A to Float
				A_Float = A[i * lda + k] * (1<<scale);
				
				//Turn into submodule
				// A_FX = (int)A_Float;
				// if(A_Float - A_FX >= 0.5){
				// 	A_FX++;
				// }
				


				FixedPointValue = ALPHA * A_FX;
				
				A_PART_TWO = (float)(FixedPointValue)/(1<<scale);
				PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
				
			
				for (j = 0; j < N; ++j) {
					B_Float = B[k * ldb + j] * (1<<scale);
					C_Float = C[i * ldc + j] * (1<<scale);
					
					//Turn into submodule
					B_FX = (int)B_Float;
					if(B_Float - B_FX >= 0.5){
						B_FX++;
					}
					
					//Turn into submodule
					C_FX = (int)C_Float;
					if(C_Float - C_FX >= 0.5){
						C_FX++;
					}

					FixedPointValue = A_PART_TWO*B_FX;
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
	SC_MODULE(conversion){
		sc_in <float>* fp_num; 
		sc_out <float>* FX;
		void convert(){
			FX = (int)fp_num;
			if(fp_num - FX >= 0.5){
				FX++;
			}
		}
		
		SC_CTOR(conversion) {
			cout<<"Converting"<<endl;
			SC_METHOD(convert);
				sensitive<<A_Float;
				sensitive<<B_Float;
				sensitive<<C_Float;
		}	
	}
};


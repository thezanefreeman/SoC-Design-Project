//HAL header
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


void hal(int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float *C, int ldc, int trigger){
	//Copy items into accelerator memory
	int i, j, k;
	int trigger;
	int send_data;

	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			send_data=A[i * lda + k];
			//send over bus
		}
	}

	for (j = 0; j < N; ++j) {
		for (k = 0; k < K; ++k) {
			send_data=B[k*ldb + j];
			//send over bus
		}
	}

	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			send_data=C[i*ldc + j];
			//send over bus
		}
	}

	trigger=1;//set trigger over bus to telling if to start gemm 
	
}

void gemm_nn(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
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


/*
Hardware Abstraction Layer
Move this to a seperate file
*/

void appGEMM(int M, int N, int K, float ALPHA,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc)
{
    fprintf(pfVStim,”halFunction1InputData1 = %d ;\n”, data1);
    //Check if the Driver is ready 
    DRV_checkReadyGEMM(TRUE); //Not Implemented Correecly
    DRV_enqueueToGemmMatrixA(A, M);
    DRV_enqueueToGemmMatrixB(B, N);
    Drv_startGEMM();
    Drv_waitGEMM();
}


/*
DRIVER 
Move this to a seperate file
*/



#define MAP_SIZE 4096UL
#define MAP_MASK (MAP_SIZE - 1)
unsigned long addr0 = 0xXXXXXXXX;  // The Starting Address for our HW Accelerator
volatile void *base;
void DRV_INIT(void)
{

    base = (unsigned int*)mmap(NULL, MAP_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, \fd, addr0 & ~MAP_MASK);// map HW physical into virtual address
    //Don't Know what PROT_READ|PROT_WRITE, or MAP_SHARED
}
int DRV_waitGEMM(int waitTillReady)
{
    int result;
    do {
        result = *((int*) base);
    } while (!result && waitTillReady) ;
    return result ;
}
int DRV_enqueueToGemmMatrixA(int *MatrixA, int MatrixASize)
{
    int i;
    for (i=0; i < MatrixASize; i++)
    {
        // stream data into accelerator local memory
        *(((int*) base)+4) = MatrixA[i];
    }
}

int DRV_enqueueToGemmMatrixB(int *MatrixB, int MatrixBSize)
{
    int i;
    for (i=0; i < MatrixBSize; i++)
    {
        // stream data into accelerator local memory
        *((((int*) base)+4)) = MatrixB[i];
    }
}


/*
Hardware Accelerator
*/


#define READ_CMD  (0x0 << 31)
#define WRITE_CMD (0x1 << 31)

#define COMMAND_MASK 0x80000000

int det_int = 0;

// signal handler for receiving events from hardware driver
void sighandler(int signo)
{
  if(signo==SIGIO)
    {
      det_int++;
      printf("\nGemm Hardware Requested\n");
    }
  return;
}


int main(){
//       // install signal handler
//   sigemptyset(&action.sa_mask);
//   sigaddset(&action.sa_mask, SIGIO);

//   action.sa_handler = sighandler;
//   action.sa_flags=0;

//   sigaction(SIGIO, &action, NULL);

  // open hardware device (driver)

    int gemm_device_driver;
  gemm_device_driver=open("/dev/gemm_accelerator", O_RDWR);
  if(fd < 0)
  {

      printf("Unable to open /dev/gemm_accelerator.  Ensure it exists!\n");
      return -1;
  }
  fcntl(fd, F_SETOWN, getpid());
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL)|O_ASYNC);
  if(argc > 1) {
    //Assign val
    val = atol(argv[1]);

    //Write to addr0
    ioctl(fd, WRITE_CMD + 0, &val);

  } else {

    //Read hardware 
    ioctl(fd, READ_CMD + 0, &result);

    printf("The SystemC time is %lu ns\n", result);

    ioctl(fd, READ_CMD + 4, &result);

    printf("The SystemC clock is %lu\n", result);
  }

  // Read interrupt
  ioctl(fd, READ_CMD + 3, &result);
  printf("Interrupt is %lu\n", result);

  // Trigger interrupt
  val = 1;
  ioctl(fd, WRITE_CMD + 3, &val);

  //Wait for interrupt
  while(!det_int) continue;

  printf("Interrupt received\n");

  //In the end, close the device driver
  close(fd);
}
//Bus interface

//Iterate through each matrix A, B, C. Send or take on bus.
//Make a slave
SC_MODULE(GEMM_IF){
	//Contains trigger signal/register that initiates GEMM process. Ideally on HAL side, gemm.c will activate trigger after passing all the memory through to the scratchpad memory
	//Will send a receive a done signal from GEMM HA. HAL will notify CPU that it is finished through interrupt or poll. In which case the CPU will start reading from scratchpad.
    //Taking values
		//Go through A
			//Read(Value from CPU,A[current])

		//Go through B
			//Read(Value from CPU,B[current])
    //Reading values
		//Go through C
			//Write(C[current], value of address)
	sc_in <bool> trigger;
	sc_in <bool> done; //notified when gemm.c finishes.
	sc_out <bool> start; //starts gemm.c module
	sc_out <bool> finished;
	sc_in <int> mat_val;
	void if_logic(){
		
		
		//see if trigger is high
		if(trigger){
			start=1;
		}
		//when computational unit sends done signal, tell cpu gemm is done
		if(done){//
			finished=1;
		}
	}
	SC_CTOR(GEMM_IF){
		SC_METHOD(if_logic);
		sensitive << trigger << mat_val;
	}
};
/*************************************/
/* Peter C. Loiacono                 */
/* Computes FFT using GPU Cuda cores */ 
/*************************************/

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "input_image.cc"
#include "complex.cc"

#define T_P_B 512

using namespace std;


/************************/
/* GPU device functions */ 
/************************/

// Orders input so even indices contained in first half
// and odd indices in second
/*__device__ void split(Complex *inData, int size) {
	Complex *cpyData = new Complex[size/2];
	for (int i=0; i < size/2; i++) {
		cpyData[i] = inData[i*2 + 1];
	}
	for (int i=0; i < size/2; i++) {
		inData[i] = inData[i*2]; 
	}
	for (int i=0; i < size/2; i++) {
		inData[i+size/2] = cpyData[i];
	}
	delete[] cpyData;
} */

// Computes Danielson-Lanczos FFT
__device__ void FFT(Complex *inData, int size) {
	if (size < 2) {
	   //End of stack, do nothing
	} else {
		//split(inData, size);

		// Split code 	
		for (int i=0; i < size/2; i++) {
			cpyData[i] = inData[i*2 + 1];
		}
		for (int i=0; i < size/2; i++) {
			inData[i] = inData[i*2]; 
		}
		for (int i=0; i < size/2; i++) {
			inData[i+size/2] = cpyData[i];
		}
		delete[] cpyData;


		FFT(inData, size/2);
		FFT(inData+size/2, size/2);

		for (int i=0; i < size/2; i++) {
			Complex even     = inData[i];
			Complex odd      = inData[i+size/2];
			Complex w        = Complex(cos(-2.*PI*i/size), sin(-2.*PI*i/size));
			inData[i]        = even + w * odd;
			inData[i+size/2] = even - w * odd; 
		}
	} 
} 


// Computes transpose of given Complex array
__device__ void transpose(Complex* inData, int size) {
	Complex* tPosed = new Complex[size];
	// Transpose values to new matrix
	int w = sqrt(size);
	for (int i=0;i<w;i++) {
		for (int j=0;j<w;j++) {
			tPosed[j*w+i] = inData[j+w*i];
		}
	}
	// Put back in original matrix 
	for (int i=0;i<w;i++) {
		for (int j=0;j<w;j++) {
			inData[i*w+j] = tPosed[i*w+j];
		}
	}
	delete [] tPosed;
}

/*******************/
/* Helper Funtions */
/*******************/

// Outputs data into text file with specified name from input
void output(char* fileName, Complex* outData, int width) {
	ofstream outFile;
	outFile.open(fileName);
	for(int i=0;i<width;i++) {
 		for(int j=0;j<width;j++) {
   			outFile << outData[i*width+j] << " ";
  		}
 		outFile << "\n";
 	}
}

int main(int argc, char* argv[]) 
{
	int width, height;
	/******************/
	/* Process inputs */
	/******************/
	if (argc != 4) {
		cout << "Please format command as: ./p33 <forward> <INPUTFILE> <OUTPUTFILE>" << endl;
	} else {
        InputImage in(argv[2]);
        width  = in.get_width();
        //cout << width << endl;
        height = in.get_height();
        //cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = in.get_image_data();

		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			//cout << i <<  " = " << data1[i] << endl;
        }

		FFT(data1, size);

		/*for (int i = 0; i < size/4; i++) {
			//cout << i <<  " = " << data1[i] << endl;
		}*/

        //data1[] contains all elements to pass to GPU for FFT
    
    	/************************/
        /* GPU Logic for 2D FFT */
		/************************/

		// Device arrays
		Complex *d_in, *d_out;
		// Local arrays
		Complex in[size], out[size];

		// Copy data into array to be used by GPU
		for (int i = 0; i < size; i++) {
			in[i] = data1[i]; 
			//cout << i <<  " = " << data1[i] << endl;
        }

		cudaMalloc((void**)&d_in, size*sizeof(Complex));
		
		// Logic for calculating 2D FFT using defined device functions
		for(int i=0; i<2; i++) {
    		if (i == 1) {
				cudaMemcpy(d_in, in, size*sizeof(Complex), cudaMemcpyHostToDevice);
				transpose<<<((size + T_P_B-1) / T_P_B), T_P_B>>>(d_in, size);
				cudaMemcpy(out, d_out, size*sizeof(Complex), cudaMemcpyHostToDevice);
    		}
    		for(int j=0;j<height;j++) {
    			cudaMemcpy(d_in, in, size*sizeof(Complex), cudaMemcpyHostToDevice);
				FFT<<<((size + T_P_B-1) / T_P_B), T_P_B>>>(d_in, size);
				cudaMemcpy(out, d_out, size*sizeof(Complex), cudaMemcpyHostToDevice);
    		}
    	}

    	output(argv[3], out, width);

    	// Garbage collection of memory
        cudaFree(d_in);
        cudaFree(d_out);
    }
	return 0;
}
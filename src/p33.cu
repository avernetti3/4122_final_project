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

#include "input_image.cuh"
#include "complex.cuh"

#define T_P_B 512

using namespace std;


/************************/
/* GPU device functions */ 
/************************/

__global__ void DFT(Complex* inData, int size) {

	Complex* H = new Complex[size];
	Complex sum;
	for (int i=0;i<size;i++) {
		sum = (0,0);
		for (int j=0;j<size;j++) {
			Complex w = Complex(cos(-2.*PI*i*j/size), sin(-2*PI*i*j/size));
			sum = sum + inData[j]*w;
		}
		H[i] = sum;
	}
	// Put values back
	for (int i=0;i<size;i++) {
		inData[i] = H[i];
	}
	delete [] H;
}


// Computes transpose of given Complex array
__global__ void transpose(Complex *inData, int size, int sqrt_val) {
	Complex* tPosed = new Complex[size];
	// Transpose values to new matrix
	int w = sqrt_val;
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

int main(int argc, char* argv[]) 
{
	int width, height;
	/******************/
	/* Process inputs */
	/******************/
	if (argc != 4) {
		cout << "Please format command as: ./p33 <forward> <INPUTFILE> <OUTPUTFILE>" << endl;
	} else {
        InputImage inImage(argv[2]);
        width  = inImage.get_width();
        //cout << width << endl;
        height = inImage.get_height();
        //cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = inImage.get_image_data();

		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			//cout << i <<  " = " << data1[i] << endl;
        }

		/*for (int i = 0; i < size/4; i++) {
			//cout << i <<  " = " << data1[i] << endl;
		}*/

        //data1[] contains all elements to pass to GPU for FFT
    
    	/************************/
        /* GPU Logic for 2D FFT */
		/************************/

		// Device arrays
		Complex *d_array;

		cudaMalloc((void**)&d_array, size*sizeof(Complex));

		// Logic for calculating 2D DFT using defined device functions
		for(int i=0; i<height; i++) {
			Complex tmp[height];

			for (int j=0; j<width; j++) {
				tmp[j] = data1[width*i + j];
			}

			cudaMemcpy(d_array, tmp, height*sizeof(Complex), cudaMemcpyHostToDevice);
			DFT<<<1,1>>>(d_array, width);
			cudaMemcpy(tmp, d_array, height*sizeof(Complex), cudaMemcpyDeviceToHost);

			for (int j=0; j<width; j++) {
				data1[width*i + j] = tmp[j];
			}
		}
		
		int sqrt_val = sqrt(size);
		cout << sqrt_val << endl;
		cudaMemcpy(d_array, data1, size*sizeof(Complex), cudaMemcpyHostToDevice);
		transpose<<<1,1>>>(d_array, size, sqrt_val);
		cudaMemcpy(data1, d_array, size*sizeof(Complex), cudaMemcpyDeviceToHost);


		
		for(int i=0; i<height; i++) {
                        Complex tmp[height];

                        for (int j=0; j<width; j++) {
                                tmp[j] = data1[width*i + j];
                        }

                        cudaMemcpy(d_array, tmp, height*sizeof(Complex), cudaMemcpyHostToDevice);
                        DFT<<<1,1>>>(d_array, width);
                        cudaMemcpy(tmp, d_array, height*sizeof(Complex), cudaMemcpyDeviceToHost);

                        for (int j=0; j<width; j++) {
                                data1[width*i + j] = tmp[j];
                        }
                }

		sqrt_val = sqrt(size);
		cudaMemcpy(d_array, data1, size*sizeof(Complex), cudaMemcpyHostToDevice);
		transpose<<<1,1>>>(data1, size, sqrt_val);
		cudaMemcpy(data1, d_array, size*sizeof(Complex), cudaMemcpyDeviceToHost);

    	inImage.save_image_data(argv[3], data1, width, height);

    	// Garbage collection of memory
        cudaFree(d_array);
    }
	return 0;
}

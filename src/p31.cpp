/******************************************/
/* Peter C. Loiacono                      */
/* Andy Vernetti                          */
/* Computes FFT using CPU multi-threading */ 
/******************************************/

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thread>

#include "input_image.h"
#include "complex.h"

#define NUM_CORES 8

using namespace std;

const float PI = 3.14159265358979f;

/*******************/
/* Helper Funtions */
/*******************/

// Regular implementation of Fourier Transform
void DFT(Complex* inData, int size) {
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

// Matrix transpose function (works)
void transpose(Complex* inData, int size) {
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


int main(int argc, char* argv[]) 
{
	int width, height;
	/******************/
	/* Process inputs */
	/******************/
	if (argc != 4) {
		cout << "Please format command as: ./p31 <forward> <INPUTFILE> <OUTPUTFILE>" << endl;
	} else {
        InputImage inImage(argv[2]);
        width  = inImage.get_width();
        //cout << width << endl;
        height = inImage.get_height();
        //cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = inImage.get_image_data();

        // Initialize variables for multithreading
        int threadsUsed, dataPerNode;
        if(height<=NUM_CORES) {
        	threadsUsed = height;
        	dataPerNode = 1;
        }
        else {
        	threadsUsed = NUM_CORES;
        	dataPerNode = height/NUM_CORES;
        }

        //data1[] contains all elements to pass to CPU threads for DFT
		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
        }

        /************************/
        /* CPU Logic for 2D DFT */
        /************************/

        int offset = width*dataPerNode;

        // Do DFT on each row with threads
        for(int i=0;i<dataPerNode;i++) {
        	thread t1(DFT, data1+width*i, width);
        	thread t2(DFT, data1+width*i+offset, width);
        	thread t3(DFT, data1+width*i+offset*2, width);
        	thread t4(DFT, data1+width*i+offset*3, width);
        	thread t5(DFT, data1+width*i+offset*4, width);
        	thread t6(DFT, data1+width*i+offset*5, width);
        	thread t7(DFT, data1+width*i+offset*6, width);
        	thread t8(DFT, data1+width*i+offset*7, width);
        	t1.join();
	        t2.join();
	        t3.join();
	        t4.join();
	        t5.join();
	        t6.join();
	        t7.join();
	        t8.join();
        }

        // Transpose data
        transpose(data1, size);

        // Do DFT on the transformed and transposed rows
        for(int i=0;i<dataPerNode;i++) {
        	thread t1(DFT, data1+width*i, width);
        	thread t2(DFT, data1+width*i+offset, width);
        	thread t3(DFT, data1+width*i+offset*2, width);
        	thread t4(DFT, data1+width*i+offset*3, width);
        	thread t5(DFT, data1+width*i+offset*4, width);
        	thread t6(DFT, data1+width*i+offset*5, width);
        	thread t7(DFT, data1+width*i+offset*6, width);
        	thread t8(DFT, data1+width*i+offset*7, width);
        	t1.join();
	        t2.join();
	        t3.join();
	        t4.join();
	        t5.join();
	        t6.join();
	        t7.join();
	        t8.join();
        }

        // Transpose back to original orientation
        transpose(data1, size);

        // Output for debugging
		for (int i = 0; i < size/4; i++) {
			cout << i <<  " = " << data1[i] << endl;
		}

		// Output to file
		inImage.save_image_data(argv[3], data1, width, height);

    }
    return 0;
}

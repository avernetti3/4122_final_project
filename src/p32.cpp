/*********************************************
Created by Andy Vernetti
12/12/18
Source Code for MPI implementation of FFT
*********************************************/

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "input_image.h"
#include "complex.h"

using namespace std;

const double EulerC = exp(1.0);
const float PI = 3.14159265358979f;

// Separate the array into two halves of the array, the first
// half being even and the second half being odd
void split(Complex* inData, int size) {
	Complex* cpyData = new Complex[size/2];
	for(int i=0;i<size/2;i++) {
		cpyData[i] = inData[i*2+1];
	}
	for(int i=0;i<size/2;i++) {
		inData[i] = inData[i*2];
	}
	for(int i=0;i<size/2;i++) {
		inData[i+size/2] = cpyData[i];
	}
	delete[] cpyData;
}

// MPI FFT using the Danielson-Lanczos Algorithm
void FFT(Complex *inData, int size) {
	if (size<2) {
		// Return at bottom of recursion
	}
	else {
		split(inData, size);
		FFT(inData, (size/2)); // FFT the even indices
		FFT(inData+(size/2), (size/2)); // FFT the odd indices

		// Recombine the data
		for(int i=0;i<size/2;i++) {
			Complex even = inData[i];
			Complex odd = inData[i+(size/2)];
			// Complex t = polar(1.0, -2*PI*i/size) * odd;
			Complex w = Complex(cos(2*PI*i/size),-1*sin(2*PI*i/size));
			inData[i] = even + w*odd;
			inData[i+(size/2)] = even - w*odd;
		}
	}
}


int main(int argc, char* argv[]) 
{
	/*
	// Initialize MPI environment
	MPI_Init(NULL, NULL);

	// Set up variables related to MPI
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	*/

	int width, height;
	/******************/
	/* Process inputs */
	/******************/
	if (argc != 4) {
		cout << "Please format command as: ./p32 <forward> <INPUTFILE> <OUTPUTFILE>" << endl;
	} 
	else {
        InputImage in(argv[2]);
        width  = in.get_width();
        cout << width << endl;
        height = in.get_height();
        cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = in.get_image_data();

		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			if (i<size){ 
				cout << i <<  " = " << data1[i] << endl;
        	}
    	}

		FFT(data1, size);

		for (int i = 0; i < size; i++) {
			cout << i <<  " = " << data1[i] << endl;
		}
        //data1[] contains all elements to pass to GPU for FFT
    
        //TODO: Write GPU Code using the processed data found above
    }

    //MPI_Finalize();

	return 0;
}

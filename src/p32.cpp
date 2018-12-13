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

// MPI FFT using the Danielson-Lanczos Algorithm (works)
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
			Complex w = Complex(cos(-2.*PI*i/size), sin(-2.*PI*i/size));
			inData[i] = even + w*odd;
			inData[i+(size/2)] = even - w*odd;
		}
	}
}


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

// Function to write out to a file
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

    	//data1[] contains all elements to use w/ MPI for FFT

    	// Do FFT on each row, then transpose and do on each column
    	for(int j=0;j<2;j++) {
    		if(j==1) {transpose(data1, size);}
    		for(int i=0;i<height;i++) {
    			FFT(data1+width*i,width);
    		}
    	}
    	transpose(data1, size);
		
	// Output for debugging
	for (int i = 0; i < size; i++) {
		cout << i <<  " = " << data1[i] << endl;
	}
		
	// Output to file
	output(argv[3], data1, width);
    
        //TODO: Write MPI Code using the processed data found above
    }

    //MPI_Finalize();

	return 0;
}

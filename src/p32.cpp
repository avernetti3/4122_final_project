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

#include "mpi.h"
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


// Regular implementation of Fourier Transform (for threads)
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
	// Initialize MPI environment
	MPI_Init(NULL, NULL);

	// Set up variables related to MPI
	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Set up MPI complex class
	MPI_Datatype mpi_complex;
	MPI_Type_contiguous(2, MPI_FLOAT, &mpi_complex);
	MPI_Type_commit(&mpi_complex);

	int width, height;
	/******************/
	/* Process inputs */
	/******************/
	if (argc != 4 && world_rank == 0) {
		cout << "Please format command as: ./p32 <forward> <INPUTFILE> <OUTPUTFILE>" << endl;
	} 
	else {
        InputImage in(argv[2]);
        width  = in.get_width();
        //cout << width << endl;
        height = in.get_height();
        //cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = in.get_image_data();

        // Set variables to use with data accessing
        int procsUsed;
        int dataPerNode;
        if (height<=world_size) {
        	procsUsed = height;
        	dataPerNode = 1;
        }
        else {
        	procsUsed = world_size;
        	dataPerNode = height/world_size;
        }

		/*
		// Debugging ouput
		if (world_rank == 0) {
			cout << "Number of processors used: " << procsUsed << "\n";
			cout << "Datapoints per proc: " << dataPerNode << "\n";
		}
		*/

		//data1[] contains all elements to use w/ MPI for FFT
		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			if (i<size){ 
				//cout << i <<  " = " << data1[i] << endl;
        	}
    	}

    	// Do FFT on each row with MPI
    	int offset = world_rank*width*dataPerNode;
    	if (world_rank<procsUsed) {
	    	for(int i=0;i<dataPerNode;i++) {
	    		FFT(data1+offset+width*i, width);
			}
		}

		// Pass data back to master rank
		for(int i=1;i<procsUsed;i++) {
			if(world_rank==i) {
				MPI_Send(&data1[offset], width*dataPerNode, mpi_complex, 0, 0, MPI_COMM_WORLD);
			}
			if(world_rank==0) {
				MPI_Recv(&data1[i*width*dataPerNode], width*dataPerNode, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}		

		MPI_Barrier(MPI_COMM_WORLD);

		// Transpose data on master rank
		if (world_rank==0) {
			transpose(data1, size);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// Pass data back out
		for(int i=1;i<procsUsed;i++) {
			if(world_rank==0) {
				MPI_Send(&data1[i*width*dataPerNode], width*dataPerNode, mpi_complex, i, 0, MPI_COMM_WORLD);
			}
			if(world_rank==i) {
				MPI_Recv(&data1[offset], width*dataPerNode, mpi_complex, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}

		// Do FFT on transformed and transposed rows
		if (world_rank<procsUsed) {
	    	for(int i=0;i<dataPerNode;i++) {
	    		FFT(data1+offset+width*i, width);
			}
		}

		// Collect data on master rank again
		for(int i=1;i<procsUsed;i++) {
			if(world_rank==i) {
				MPI_Send(&data1[offset], width*dataPerNode, mpi_complex, 0, 0, MPI_COMM_WORLD);
			}
			if(world_rank==0) {
				MPI_Recv(&data1[i*width*dataPerNode], width*dataPerNode, mpi_complex, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
		}	

		MPI_Barrier(MPI_COMM_WORLD);

		// Transpose data back to original orientation
		if (world_rank==0) {
			transpose(data1, size);
		}

		MPI_Barrier(MPI_COMM_WORLD);
    	
		// Output for debugging
		/*
		if (world_rank==0) {
			for (int i = 0; i < size; i++) {
				cout << i <<  " = " << data1[i] << endl;
			}
		}
		*/

		// Output to file (updated to use given function)
		if(world_rank==0) {in.save_image_data(argv[3], data1, width, height);}
    
    }

    MPI_Finalize();

	return 0;
}

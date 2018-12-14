/******************************************/
/* Peter C. Loiacono                      */
/* Computes FFT using CPU multi-threading */ 
/******************************************/

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <thread>

#include "input_image.cc"
#include "complex.cc"

#define NUM_CORES 8

using namespace std;

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

		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			//cout << i <<  " = " << data1[i] << endl;
        }

		/*for (int i = 0; i < size/4; i++) {
			//cout << i <<  " = " << data1[i] << endl;
		}*/

        //data1[] contains all elements to pass to CPU threads for DFT

        /************************/
        /* CPU Logic for 2D DFT */
        /************************/

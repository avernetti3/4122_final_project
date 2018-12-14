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

const double EulerConstant = exp(1.0);

//TODO: Debug
/*__device__ */ void order(Complex *inData, int size) {
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
}

//TODO: Debug currently doesn't compute properly
/*__device__*/ void FFT(Complex *inData, int size) {
	if (size < 2) {
	
	} else {
		order(inData, size);
		FFT(inData, size/2);
		FFT(inData+size/2, size/2);

		for (int i=0; i < size/2; i++) {
			Complex even     = inData[i];
			Complex odd      = inData[i+size/2];
			Complex w        = Complex(pow(EulerConstant, 0), pow(EulerConstant,-2.*PI*i/size));
			inData[i]        = even + w * odd;
			inData[i+size/2] = even + w * odd; 
		}
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
        cout << width << endl;
        height = in.get_height();
        cout << height << endl;

        int size = width*height;
        Complex data1[size];
        Complex* data2 = in.get_image_data();

		for (int i = 0; i < size; i++) {
			data1[i] = data2[i]; 
			//cout << i <<  " = " << data1[i] << endl;
        }

		FFT(data1, size);

		for (int i = 0; i < size/4; i++) {
			//cout << i <<  " = " << data1[i] << endl;
		}
        //data1[] contains all elements to pass to GPU for FFT
    
        //TODO: Write GPU Code using the processed data found above
    }
	return 0;
}
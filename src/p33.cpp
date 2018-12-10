#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "input_image.cc"
#include "complex.cc"

#define T_P_B 512

using namespace std;

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
        	cout << i <<  " = " << data1[i] << endl;
        }

        //data1[] contains all elements to pass to GPU for FFT
    
        //TODO: Write GPU Code using the processed data found above
    }
	return 0;
}
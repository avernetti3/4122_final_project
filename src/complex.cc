//
// Created by brian on 11/20/18.
//

#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

Complex Complex::operator+(const Complex &b) const {
 Complex a;
 a.real = this->real + b.real;
 a.imag = this->imag + b.imag;
 return a;
}

Complex Complex::operator-(const Complex &b) const {
 Complex a;
 a.real = this->real - b.real;
 a.imag = this->imag - b.imag;
 return a;
}

Complex Complex::operator*(const Complex &b) const {
 Complex a;
 a.real = (this->real)*b.real - (this->imag)*b.imag;
 a.imag = (this->real)*b.imag + (this->imag)*b.real;
 return a;
}

Complex Complex::mag() const {
 Complex a;
 float x2 = pow((this->real),2);
 float y2 = pow((this->imag),2);
 a.real = sqrt(x2+y2),(1/2);
 return a;
}

Complex Complex::angle() const {
 Complex a;
 float x = this->real;
 float y = this->imag;

 if(x>0) {a.real = atan(y/x);}
 else if(x<0 && y>=0) {a.real = atan(y/x)+PI;}
 else if(x<0 && y<0) {a.real = atan(y/x)-PI;}
 else if(x==0 && y>0) {a.real = PI/2;}
 else if(x==0 && y<0) {a.real = -1*PI/2;}
 else {a.real=0;}
 a.imag=0;
 return a;
}

Complex Complex::conj() const {
 Complex a;
 a.real = this->real;
 a.imag = -1*(this->imag);
 return a;
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}

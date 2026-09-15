//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 1.0
// Date:    1998
// Author:  Rohin Gosling
//
// Description:
//
//   Declaration of the MathAI class, a collection of general-purpose static mathematical functions used throughout the
//   Common Artificial Intelligence Library. The class groups the activation functions and their first derivatives
//   required by feed-forward neural networks (linear, sigmoid, ReLU, hyperbolic tangent, and others) together with a
//   small set of scalar and vector utilities. Every member is static, so MathAI is used as a function library rather
//   than being instantiated.
//
//---------------------------------------------------------------------------------------------------------------------

#ifndef MATH_ANN_H
#define MATH_ANN_H

#include <vector>

using namespace std;

//*********************************************************************************************************************
// Class: MathAI
//
// Description:
//
//   A stateless library of static mathematical functions for artificial intelligence and machine-learning code. MathAI
//   provides the activation functions and their first derivatives used during forward and backward propagation, a set
//   of supporting scalar functions (signum, unit step, hyperbolic secant), and vector helpers (element sum and
//   Hadamard product). It also defines the Vector, Matrix, and Tensor type aliases shared across the library. Because
//   every member is static, MathAI is never instantiated -- call its functions through the class name, for example
//   MathAI::Sigmoid ( x ).
//
//*********************************************************************************************************************

class MathAI
{
    public:

    // Data types.

    //=================================================================================================================
    // Nested Types
    //=================================================================================================================

    typedef vector <double>                    Vector;
    typedef vector <vector <double>>           Matrix;
    typedef vector <vector < vector <double>>> Tensor;

    //=================================================================================================================
    // Methods
    //=================================================================================================================

    static double Linear               ( double x );
    static double LinearDerivative     ( double x );
    static double Linear               ( double x, double a );
    static double LinearDerivative     ( double x, double a );
    static double Sigmoid              ( double x );
    static double sigmoid              ( double x, double k );
    static double SigmoidDerivative    ( double x );
    static double SigmoidDerivative    ( double x, double k );
    static double ReLU                 ( double x );
    static double ReLUDerivative       ( double x );
    static double Tanh                 ( double x );
    static double TanhDerivative       ( double x );
    static double Sinusoidal           ( double x );
    static double SinusoidalDerivative ( double x );
    static double sech                 ( double x );
    static double Sgn                  ( double x );
    static double Step                 ( double x );
    static double VectorSum            ( const Vector& v );
    static Vector HadamardProduct      ( const Vector& u, const Vector& v );
    static double GaussianDistribution ();
};

#endif // MATH_ANN_H

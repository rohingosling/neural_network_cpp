//--------------------------------------------------------------------------------------------------
// Project:	    N/A
// Application: All AI and Machine Learning applications.
// Class:	    AIMath
// Version:	    1.0
// Date:	    2015
// Author:	    Rohin Gosling
//
// Description:
//
//   Static math functions for use in AI and machine learning applications.
//
//--------------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <math.h>

using namespace std;

class AIMath
{
	//--------------------------------------------------------------------------------------------------
	// Constants
	//--------------------------------------------------------------------------------------------------

public:

	// ...

private:
	
	// ...

	//--------------------------------------------------------------------------------------------------
	// Data type aliases
	//--------------------------------------------------------------------------------------------------

public:

	// ...

	//--------------------------------------------------------------------------------------------------
	// Functions.
	//--------------------------------------------------------------------------------------------------

public:

	// Constructors and destructors

	AIMath  ( void );
    ~AIMath ( void );

	// Math functions.

    static double          sigmoid               ( double x );                              // Sigmoid function.
    static double          sigmoid               ( double x, double k );                    // Parametrized sigmoid function.
    static double          sigmoid_dxyd          ( double x );					            // Derivitive of a parametrized sigmoid function.
    static double          sgn                   ( double x );                              // Sign function.
    static double          step                  ( double x );                              // Step function.
    static double          sech                  ( double x );                              // Hyperbolic secant.
    static double          vector_sum            ( vector <double> v );                     // Vector sum.
    static vector <double> hadamard_product      ( vector <double> u, vector <double> v );	// Hadamard vector product.
    static double          gaussian_distribution ();										// Gaussian distribution; 0 ≤ r ≤ 1.
    		
private:

    // ...

    //--------------------------------------------------------------------------------------------------
	// Accessors and mutators
	//--------------------------------------------------------------------------------------------------

	// Accessors

	// ...

	// Mutators.

	// ...

	//--------------------------------------------------------------------------------------------------
	// Member data.
	//--------------------------------------------------------------------------------------------------

public:

	// ...
	
private:

	// ...
};


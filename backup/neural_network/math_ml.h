//--------------------------------------------------------------------------------------------------------------------------------------
// Project:		Machine Learning Platform
// Application:	Machine Learning Math Library.
// Class:       MathML
// Version:		1.0
// Date:        2015-07-10
// Author:		Rohin Gosling
//
// Description:
//
//   Mathematics functions for machine learning.
//
// Apendix A - DLL Managment
//
//   Create a DLL:
//
//     1. Set target configuration type to DLL.
//        Properties\Configuration Properties\General\Configuration Type = “Dynamic Library (.dll)
//     
//     2. Add a namespace to the DLL class.
//     
//     3. Add export symbol to compiler command line.
//        e.g. For the export symbol, “CLASS_NAME_EXPORTS”, then;
//        Properties\Configuration Properties\C/C++\Command Line\Additional Options = “/D CLASS_NAME_EXPORTS”
//     
//     4. Add a #define to the DLL header file, for the export symbol.
//        e.g. For the export symbol, “CLASS_NAME_EXPORTS”, then;
//     
//        #ifdef CLASS_NAME_EXPORTS
//            #define CLASS_NAME_API __declspec ( dllexport ) 
//        #else
//            #define CLASS_NAME_API __declspec ( dllimport ) 
//        #endif
//     
//     5. Apply the export symbol modifier to either the entire class, or selected members.
//
//     6. Specify explicate STL template classes to export before class decleration, where required.
//        e.g. template class __declspec ( dllexport ) std::vector<string>;
//             or
//        e.g. template class CLASS_NAME_API std::vector<string>;
//
//
//   Reference a DLL (Test):
//   
//     1. Create a test application.
//   
//     2. Add a reference to the DLL.
//        Properties\Common Properties\Add New Reference
//   
//     3. Add the DLL’s header file directory to the project.
//        Properties\Configuration Properties\VC++ Directories
//   
//     4. Optional: Add “using namespace” to the consumer class, for neater referencing.
//   
//
//   Reference a DLL (Implicate Linking):
//   
//     1. Create DLL comsumer application.
//
//     2. Copy both the DLL's .dll and .lib files, in the output folder.
//
//     3. Copy the DLL's header file to the source code folder.
//   
//     4. Add a reference to the LIB file.
//        Properties\Configuration Properties\Linker\Input\Aditional Dependencies = file_name.lib 
//   
//     4. Optional: Add “using namespace” to the consumer class, for neater referencing.
//
//
//   Reference a DLL (Explicate Linking):
//
//     1. TODO:
//
//     2. TODO:
//
//--------------------------------------------------------------------------------------------------

#pragma once

#ifdef MATH_ML_EXPORTS
	#define MATH_ML_API __declspec ( dllexport ) 
#else
	#define MATH_ML_API __declspec ( dllimport ) 
#endif

#include "stdafx.h"

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <math.h>

using namespace std;

namespace common
{
	class MATH_ML_API MathML
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

		MathML ( void );
		~MathML ( void );

		// Optimized functions.
				
		static double          fast_sqrt                ( double x );
		static double          fast_exp                 ( double x );

		// Neural network activation functions and their derivitives.

		// Activation functions: Threshold functions.

		static double          step                     ( double x );                              // Step function.
		static double          sgn                      ( double x );                              // Sign function.		

		// Activation functions: Linear units.
													   
		static double          relu                     ( double x );								// Hard rectified linear unit.		
		static double          prelu                    ( double x, double k );						// Paremerized hard rectified linear unit.		
		static double          dydx_relu                ( double x );								// Derivitive of rectified linear unit.
		static double          dydx_prelu               ( double x, double k );						// Derivitive of parameterized rectified linear unit.

		// Activation functions: Sigmoid (Logistic) function.

		static double          sigmoid                  ( double x );                              // Sigmoid function.
		static double          sigmoid                  ( double x, double k );                    // Parametrized sigmoid function.
		static double          dydx_sigmoid             ( double x );					           // Derivitive of a sigmoid function.
		static double          dydx_sigmoid             ( double x, double k );					   // Derivitive of a parametrized sigmoid function.		

		//  Activation functions: Trgonometric activation functons.		

		static double          tanh_normal              ( double x );                               // Normalized hyerbolic tangent.
		static double          tanh_normal              ( double x, double k );                     // Parameterized normalized hyerbolic tangent.		
		static double          sin_normal               ( double x);                                // Normalized sinusoid.		
		static double          dydx_tanh_normal         ( double x );                               // Derivative of normalized hyerbolic tangent.
		static double          dydx_tanh_normal         ( double x, double k );                     // Derivative of parameterized normalized hyerbolic tangent.
		static double          dydx_sin_normal          ( double x);                                // Normalized sinusoid.		
		
		// Activation functions: Softplus function.
													   
		static double          softplus                 ( double x );								// Soft rectified linear unit.		
		static double          softplus                 ( double x, double k, double a );			// Parameterized Soft rectified linear unit.
		static double          softplus_normal          ( double x, double k );						// Normalized Parameterized Soft rectified linear unit.
		static double          dydx_softplus            ( double x );								// Derivitive  of soft rectified linear unit.
		static double          dydx_softplus            ( double x, double k, double a );			// Parameterized derivitive of soft rectified linear unit.		
		static double          dydx_softplus_normal     ( double x, double k );						// Normalized Derivitive of Parameterized soft rectified linear unit.

		// Activation Function: Softsign.

		static double          softsign                 ( double x );								// Soft rectified linear unit.		
		static double          dydx_softsign            ( double x );								// Derivitive  of soft rectified linear unit.


		// Activation functions: Polynomial activation functions.
		
		static double          parabolic                ( double x );								// Parabolic activation function.		
		static double          quadratic                ( double x );								// Quadratic activation function.		
		static double          quadratic_transpose      ( double x );								// Transposed quadratic activation function.
		static double          cubic                    ( double x );								// Cubic activation function.		
		static double          pythagorean              ( double x );								// Pythagorean (counter-circular) activation function.		
		static double          dydx_parabolic           ( double x );								// Parabolic activation function.		
		static double          dydx_quadratic           ( double x );								// Derivitave of quadratic activation function.		
		static double          dydx_cubic               ( double x );								// Derivitave of cubic activation function.		
		static double          dydx_pythagorean         ( double x );								// Derivitave of pythagorean activation function.		
		static double          dydx_quadratic_transpose ( double x );								// Derivitave of transposed quadratic activation function.
		
		// Vector functions.

		static double          vector_sum               ( vector <double> v );						// Vector sum.
		static vector <double> hadamard_product         ( vector <double> u, vector <double> v );	// Hadamard vector product.

		// Distribution functions.

		static double          gaussian_distribution    ();											// Gaussian distribution; 0 ≤ r ≤ 1.

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
}
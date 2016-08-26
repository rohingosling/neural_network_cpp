//--------------------------------------------------------------------------------------------------
// Project:	N/A
// Application: All AI and Machine Learning applications.
// Class:	AIMath
// Version:	1.0
// Date:	2015
// Author:	Rohin Gosling
//
// Description:
//
//   Static math functions for use in AI and machine learning applications.
//
//--------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "ai_math.h"

//--------------------------------------------------------------------------------------------------
// Constructor 1/1
//--------------------------------------------------------------------------------------------------

AIMath::AIMath(void)
{
}

//--------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------

AIMath::~AIMath(void)
{
}

//--------------------------------------------------------------------------------------------------
//
//                 1
//           ────────────
//  f(x)  =  ┌       -x ┐
//           │  1 + e   │
//           └          ┘ 
//
//--------------------------------------------------------------------------------------------------

double AIMath::sigmoid ( double x )
{
    // Compute the sigmoid function.
        
    double ex = exp ( -x );             // Raise e to the power of -x. 
    double y  = 1 / ( 1 + ex );         // Calculate the sigmoid function.
        
    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
//
//                  1
//           ──────────────
//  f(x)  =  ┌       -k*x ┐
//           │  1 + e     │
//           └            ┘  
//
//--------------------------------------------------------------------------------------------------

double AIMath::sigmoid ( double x, double k )
{
    // Compute the sigmoid function.
        
    double t  = -k*x;                   // Exponent.           
    double ex = exp ( t );              // Raise e to the power of t. 
    double y  = 1 / ( 1 + ex );         // Calculate the sigmoid function.
        
    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
// 
//                          x
//                         e
//            dx     ─────────────
//  f'(x)  =  ──  =  ┌   x      ┐2
//            dy     │  e  + 1  │
//                   └          ┘  
//
//--------------------------------------------------------------------------------------------------

double AIMath::sigmoid_dxyd ( double x )
{
    // Compute the derivitive of the sigmoid function.
                      
    double d    = exp ( x );
    double dx   = ( d + 1 );
    double dxdy = d / dx*dx;            // Calculate the derivative of the sigmoid function.
        
    // Return the result.
        
    return dxdy;
}

//--------------------------------------------------------------------------------------------------
//
//  y  =  sgn(x)
//
//--------------------------------------------------------------------------------------------------

double AIMath::sgn ( double x )
{
	// Declare local varaibles.

    double y = 0.0;
        
    // Compute the sign function.
        
    if ( x > 0.0  ) y =  1.0;
    if ( x < 0.0  ) y = -1.0;
    if ( x == 0.0 ) y =  0.0;
        
    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
//
//  y  =  step(x)
//
//--------------------------------------------------------------------------------------------------

double AIMath::step ( double x )
{
	// Declare local varaibles.

    double y = 0.0;
        
    // Compute the sign function.
        
    if ( x >  0.0  ) y =  1.0;
    if ( x <= 0.0  ) y =  0.0;        
        
    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
//
//           1
//  y  =  ───────  
//        cosh(x)
//
//--------------------------------------------------------------------------------------------------
              
double AIMath::sech ( double x )
{
    // Compute hyperbolic secant.
        
    double y = 1.0 / cosh ( x );
        
    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
//
//    _      n
//  S(v)  =  ∑ ( vᵢ )
//          i=0 
//
//--------------------------------------------------------------------------------------------------


double AIMath::vector_sum ( vector <double> v )
{
	// Declare local varaibles.

    double sum = 0.0;
        
    // Calculate sum.
        
    for ( double x : v )
    {
        sum += x;
    }
        
    // Return the result.
        
    return sum;
}

//--------------------------------------------------------------------------------------------------
//  _    _   _
//  p  = u · v
//
//--------------------------------------------------------------------------------------------------

vector <double> AIMath::hadamard_product ( vector <double> u, vector <double> v )
{
	// Declare local variables.

    vector <double> hp;  
    double          p;
    long            u_size = u.size();
    long            v_size = v.size();

    // Compute the optimal array size.
    // - For safty, take the smallest one. But they 'should' both be equal size.

    int s = (u_size < v_size) ? u_size : v_size;
        
    // Make sure the two vectors are the same size.

    if ( u_size == v_size )
    {       
        // Calculate the Hadamard product of vectors u and v.    
            
        for ( int i = 0; i < s; i++ )
        {
            p = u[i] * v[i];

            hp.push_back ( p );
        }
    }
        
    // Return the result. 
        
    return hp;
}

//--------------------------------------------------------------------------------------------------
//
//    x  =  rnd ( 0 ≤ x ≤ 1 )
//
//
//          1     1         3
// f(x)  =  ─  +  ─ (2x - 1)
//          2     2
//
//--------------------------------------------------------------------------------------------------

double AIMath::gaussian_distribution ()
{
    // Initialize random number generator.
    
    random_device                      rd;
    mt19937                            random_generator    ( rd() );
    uniform_real_distribution <double> random_distribution ( 0.0, 1.0 );

    // Note:
	//
    // - This was ported from my Java version, and should be upgraded to take advantage of the
    //   C++11 random number library.
	//
    // - In C++11, we can generate a Gaussian distribution by using 'std::normal_distribution' from 
    //   the random number library.    

    // Compute gaussian distribution.

    double x        = random_distribution ( random_generator ); 
	double base     = 2 * x - 1;    
	double exponent = 3;
    double y        = 0.5 + 0.5 * pow ( base, exponent );

    // Return the result.
        
    return y;
}

//--------------------------------------------------------------------------------------------------


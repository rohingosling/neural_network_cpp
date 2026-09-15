//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 1.0
// Date:    1998
// Author:  Rohin Gosling
//
// Description:
//
//   Implementation of the MathAI class. Defines the static mathematical functions declared in math_ai.h: the
//   activation functions and their first derivatives used during forward and backward propagation, a set of supporting
//   scalar functions (signum, unit step, hyperbolic secant), and the vector helpers (element sum and Hadamard
//   product). Every function is stateless and is called through the class name.
//
//---------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <cmath>
#include <vector>
#include <random>

#include "math_ai.h"

using namespace std;

//=====================================================================================================================
// Methods
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
// Method: Sigmoid
//
// Description:
//
//   Logistic sigmoid activation function. Squashes any real input into the open interval (0, 1), producing the smooth,
//   S-shaped response used for binary-classification outputs and as a classic hidden-layer activation.
//
//        1
//     ───────
//          -x
//     1 + ℯ
//
// Arguments:
//
//   x (double):
//     The pre-activation value (weighted sum) to squash.
//
// Returns:
//
//   double:
//     A value in the open interval (0, 1).
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Sigmoid ( double x )
{
    double ex = exp ( -x );
    double y  = 1.0 / (1.0 + ex);

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: sigmoid
//
// Description:
//
//   Parametric logistic sigmoid. Adds a steepness parameter k that controls how sharply the curve transitions through
//   its midpoint. A larger k approaches a hard step, while a smaller k flattens the response.
//
//         1
//     ─────────
//          -k⋅x
//     1 + ℯ
//
// Arguments:
//
//   x (double):
//     The pre-activation value to squash.
//
//   k (double):
//     The steepness (gain) of the sigmoid curve.
//
// Returns:
//
//   double:
//     A value in the open interval (0, 1).
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::sigmoid ( double x, double k )
{
    // Compute the sigmoid function.

    double ex = exp ( -k * x );
    double y  = 1.0 / ( 1.0 + ex );

    // Return the result.

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: SigmoidDerivative
//
// Description:
//
//   First derivative of the logistic sigmoid, used when back-propagating error gradients through a sigmoid activation.
//   The slope is largest near x = 0 and approaches zero as the input saturates towards either extreme.
//
//         x
//        ℯ
//     ─────────
//             2
//     ⎛ x    ⎞
//     ⎝ℯ  + 1⎠
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated.
//
// Returns:
//
//   double:
//     The slope of the sigmoid at x.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::SigmoidDerivative ( double x )
{
    // Compute the derivitive of the sigmoid function.

    double ex   = exp ( x );
    double ex1  = ( ex + 1.0 );
    double dydx = ex / ex1 * ex1;

    // Return the result.

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: SigmoidDerivative
//
// Description:
//
//   First derivative of the parametric logistic sigmoid with steepness k, used when back-propagating gradients through
//   a parametric sigmoid activation.
//
//          k⋅x
//       k⋅ℯ
//     ───────────
//               2
//     ⎛ k⋅x    ⎞
//     ⎝ℯ    + 1⎠
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated.
//
//   k (double):
//     The steepness (gain) of the underlying sigmoid curve.
//
// Returns:
//
//   double:
//     The slope of the parametric sigmoid at x.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::SigmoidDerivative ( double x, double k )
{
    // Compute the derivitive of the sigmoid function.

    double ex   = exp ( k * x );
    double kex  = k * ex;
    double ex1  = ex + 1.0;
    double dydx = kex / ex1 * ex1;

    // Return the result.

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ReLU
//
// Description:
//
//   Rectified Linear Unit activation function. Passes positive inputs through unchanged and clamps negative inputs to
//   zero. ReLU is the default hidden-layer activation in this library because it is cheap to evaluate and lets
//   gradients flow without saturating for positive inputs.
//
//     ⎧x  for x > 0
//     ⎨
//     ⎩0  otherwise
//
// Arguments:
//
//   x (double):
//     The pre-activation value to rectify.
//
// Returns:
//
//   double:
//     x when x > 0, otherwise 0.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::ReLU ( double x )
{
    double y = ( x > 0.0 ) ? x : 0.0;

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ReLUDerivative
//
// Description:
//
//   First derivative of the Rectified Linear Unit. The slope is one for positive inputs and zero elsewhere, which is
//   what allows ReLU gradients to pass through unattenuated during back-propagation.
//
//     ⎧1  for x > 0
//     ⎨
//     ⎩0  otherwise
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated.
//
// Returns:
//
//   double:
//     1.0 when x > 0, otherwise 0.0.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::ReLUDerivative ( double x )
{
    double dydx = ( x > 0 ) ? 1.0 : 0.0;

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Tanh
//
// Description:
//
//   Hyperbolic tangent activation function. Squashes any real input into the open interval (-1, 1), giving a
//   zero-centred, S-shaped response that often trains faster than the logistic sigmoid in hidden layers.
//
//     f(x) = tanh(x)
//
// Arguments:
//
//   x (double):
//     The pre-activation value to squash.
//
// Returns:
//
//   double:
//     A value in the open interval (-1, 1).
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Tanh ( double x )
{
    double y = tanh ( x );

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: TanhDerivative
//
// Description:
//
//   First derivative of the hyperbolic tangent activation, used when back-propagating gradients through a tanh
//   activation.
//
//     f'(x) = 1 - tanh(x)^2
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated.
//
// Returns:
//
//   double:
//     The slope of tanh at x.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::TanhDerivative ( double x )
{
    double th   = tanh ( x );
    double dydx = 1.0 - th * th;

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Linear
//
// Description:
//
//   Linear (identity) activation function. Returns its input unchanged. It is typically applied to the output layer of
//   a regression network, where a continuous, unbounded result is required.
//
//     f(x) = x
//
// Arguments:
//
//   x (double):
//     The value to pass through the linear function.
//
// Returns:
//
//   double:
//     The input value x, returned unchanged.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Linear ( double x )
{
    double y = x;

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: LinearDerivative
//
// Description:
//
//   First derivative of the Linear activation function. The derivative of the identity function is constant, so the
//   result is always one, regardless of the input.
//
//     f'(x) = 1
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated. The result does not depend on this value.
//
// Returns:
//
//   double:
//     The constant value 1.0.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::LinearDerivative ( double x )
{
    double dydx = 1.0;

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Linear
//
// Description:
//
//   Parametric linear activation function. Scales its input by the slope a, producing a straight line through the
//   origin with gradient a.
//
//     f(x) = a * x
//
// Arguments:
//
//   x (double):
//     The value to scale.
//
//   a (double):
//     The slope (gradient) applied to x.
//
// Returns:
//
//   double:
//     The product a * x.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Linear ( double x, double a )
{
    double y = a * x;

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: LinearDerivative
//
// Description:
//
//   First derivative of the parametric linear activation function. The gradient of a * x is constant, so the slope a
//   is returned regardless of the input.
//
//     f'(x) = a
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated. The result does not depend on this value.
//
//   a (double):
//     The slope of the parametric linear function, returned as the derivative.
//
// Returns:
//
//   double:
//     The slope a.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::LinearDerivative ( double x, double a )
{
    double dydx = a;

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Sinusoidal
//
// Description:
//
//   Truncated sine activation function. Behaves as sin(x) within the interval [-2*pi, 2*pi] and saturates to -1 below
//   that interval and +1 above it, giving a bounded response that is periodic within range.
//
//     ⎧  -1    for x < -2⋅π
//     ⎪
//     ⎨sin(x)  for x ≤ 2⋅π
//     ⎪
//     ⎩  1      otherwise
//
// Arguments:
//
//   x (double):
//     The pre-activation value to transform.
//
// Returns:
//
//   double:
//     sin(x) for -2*pi <= x <= 2*pi, -1 for x < -2*pi, and +1 for x > 2*pi.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Sinusoidal ( double x )
{
    const double pi  = 3.14159265358979323846;
    double       pi2 = 2 * pi;
    double       y   = 0.0;

    if ( ( x >= -pi2 ) && ( x <= pi2 ) )
    {
        y = sin ( x );
    }
    else if ( x < -pi2 )
    {
        y = -1.0;
    }
    else if ( x > pi2 )
    {
        y = 1.0;
    }

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: SinusoidalDerivative
//
// Description:
//
//   First derivative of the truncated sine activation function. Returns cos(x) within the interval [-2*pi, 2*pi] and
//   -1 outside it, matching the saturating behaviour of the function itself.
//
//     ⎧  -1    for x < -2⋅π
//     ⎪
//     ⎨cos(x)  for x ≤ 2⋅π
//     ⎪
//     ⎩  -1     otherwise
//
// Arguments:
//
//   x (double):
//     The point at which the derivative is evaluated.
//
// Returns:
//
//   double:
//     cos(x) for -2*pi <= x <= 2*pi, and -1 otherwise.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::SinusoidalDerivative ( double x )
{
    const double pi   = 3.14159265358979323846;
    double       pi2  = 2 * pi;
    double       dydx = ( ( x >= -pi2 ) && ( x <= pi2 ) ) ? cos ( x ) : -1.0;

    return dydx;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: sech
//
// Description:
//
//   Hyperbolic secant. The reciprocal of the hyperbolic cosine, giving a smooth, bell-shaped curve that peaks at x =
//   0.
//
//        1
//     ───────
//     cosh(x)
//
// Arguments:
//
//   x (double):
//     The value at which the hyperbolic secant is evaluated.
//
// Returns:
//
//   double:
//     The hyperbolic secant of x, a value in the half-open interval (0, 1].
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::sech ( double x )
{
    // Compute hyperbolic secant.

    double y = 1.0 / cosh ( x );

    // Return the result.

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Sgn
//
// Description:
//
//   Signum (sign) function. Reports the sign of its input as a discrete value, and is useful as a hard threshold or
//   for extracting the direction of a quantity.
//
//     ⎧-1  for x < 0
//     ⎪
//     ⎨0   for x = 0
//     ⎪
//     ⎩1   for x > 0
//
// Arguments:
//
//   x (double):
//     The value whose sign is reported.
//
// Returns:
//
//   double:
//     -1 when x < 0, 0 when x == 0, and +1 when x > 0.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Sgn ( double x )
{
    // Declare local variables.

    double y = 0.0;

    // Compute the sign function.

    if ( x >  0.0 ) y =  1.0;
    if ( x <  0.0 ) y = -1.0;
    if ( x == 0.0 ) y =  0.0;

    // Return the result.

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Step
//
// Description:
//
//   Unit (Heaviside) step function. Returns one for strictly positive inputs and zero otherwise, modelling a hard
//   activation threshold.
//
//     ⎧1  for x > 0
//     ⎨
//     ⎩0  otherwise
//
// Arguments:
//
//   x (double):
//     The value compared against the threshold of zero.
//
// Returns:
//
//   double:
//     1.0 when x > 0, otherwise 0.0.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::Step ( double x )
{
    // Declare local varaibles.

    double y = 0.0;

    // Compute the sign function.

    if ( x >  0.0 ) y = 1.0;
    if ( x <= 0.0 ) y = 0.0;

    // Return the result.

    return y;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: VectorSum
//
// Description:
//
//   Sums every element of a vector and returns the scalar total. Used wherever a reduction over a vector of values is
//   required, such as accumulating weighted inputs.
//
//     n - 1
//      ___ ╲
//       ╲ ╱ vᵢ
//      ╱ ‾‾‾
//     i = 0
//
// Arguments:
//
//   v (const Vector&):
//     The vector whose elements are summed.
//
// Returns:
//
//   double:
//     The sum of all elements of v. Returns 0 for an empty vector.
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::VectorSum ( const Vector& v )
{
    // Declare local variables.

    double sum = 0.0;

    // Calculate sum.

    for ( double x : v )
    {
        sum += x;
    }

    // Return the result.

    return sum;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: HadamardProduct
//
// Description:
//
//   Element-wise (Hadamard) product of two equally sized vectors. Multiplies corresponding elements to produce a new
//   vector of the same length.
//
//     p[i] = u[i] * v[i]
//
// Arguments:
//
//   u (const Vector&):
//     The first operand vector.
//
//   v (const Vector&):
//     The second operand vector. Must be the same length as u.
//
// Returns:
//
//   Vector:
//     A vector whose i-th element is u[i] * v[i]. Returns an empty vector if the operands differ in length.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Vector MathAI::HadamardProduct ( const Vector& u, const Vector& v )
{
    // Declare local variables.

    Vector hp;
    double p;
    size_t u_size = u.size ();
    size_t v_size = v.size ();

    // Make sure the two vectors are the same size.

    if ( u_size == v_size )
    {
        // Calculate the Hadamard product of vectors u and v.

        size_t s = u_size;    // We can only rach this point in the code, if `u_size == v_size`. So we can pick any one of the vector sizes.

        for ( size_t i = 0; i < s; ++i )
        {
            p = u [ i ] * v [ i ];

            hp.push_back ( p );
        }
    }
    else
    {
        // If the vectors are not of the same size, handle the error.
        // - In our case we are going to write and error message and return an empty vector.
        // - The vector will be empty because at this point we have not assigned it a value yet.

        cout << endl << "[HadamardProduct] ERROR: Vectors must be of the same size." << endl;
    }

    // Return the result.

    return hp;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: GaussianDistribution
//
// Description:
//
//   Draws a pseudo-random sample shaped by a cubic transform of a uniform variate. The transform biases samples
//   towards the centre of the [0, 1] interval, approximating a bell-shaped distribution.
//
//              3
//     (2⋅x - 1)    1
//     ────────── + ─
//         2 2
//
// Returns:
//
//   double:
//     A pseudo-random value in the interval [0, 1].
//
//---------------------------------------------------------------------------------------------------------------------

double MathAI::GaussianDistribution ()
{
    // Initialize random number generator.

    random_device                      rd;
    mt19937                            random_generator    ( rd () );
    uniform_real_distribution <double> random_distribution ( 0.0, 1.0 );

    // Compute gaussian distribution.

    double x        = random_distribution ( random_generator );
    double base     = 2.0 * x - 1;
    double y        = ( 1.0 + base * base * base ) / 2.0;

    // Return the result.

    return y;
}


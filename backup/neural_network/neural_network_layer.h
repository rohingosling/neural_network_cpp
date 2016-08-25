//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Project:	    N/A
// Application: Neural Network Test
// Class:	    NeuralNetwork
// Version:	    1.0
// Date:	    2015
// Author:	    Rohin Gosling
//
// Description:
//
//   Neural network class.
//
//   Note:
//   - Ported from my Java version of this class.
//
// Terminology:
//
// - Unit
//   + Refers to a single neural unit.
//   + Other names used in literature include; neuron, perceptron, perceptron unit, or in the
//     context of vectors, an element.
//
// - Input and Output Vectors
//   Vectors who's elements represent individual unit values (perceptron values).
//
// - Weight Vector
//   A vector who's elements represent the weights of all input dendrites to a single
//   unit (perceptron, neuron, etc).
//
// - Layer Operations
//   Calculations that carry and transform the input vector into a layer, through the various
//   stages that lead to the output vector.
//   + Weight Function.
//   + Network Function, or Net Function.
//   + Transfer Function.
//
// - Weight Function
//   + Function used to compute the relationship between the input vector and the weight vector.
//   + This is most commonly performed by calculating the Hadamard Product of the input and weight
//     vectors.
//
// - Network Function (Net Function)
//   + The network function integrates all the inputs and bias values, into a single scalar value 
//     to be used as the input into the transfer function.
//   + This is typically performed by summing up all the results of the weight function and the bias
//     unit (neuron).
//
// - Activation Function
//   The activation function defines the transfer threshold behaviour of a perceptron unit (neuron).
//
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
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

// DLL managment.

#ifdef NEURAL_NETWORK_LAYER_EXPORTS
	#define NEURAL_NETWORK_LAYER_API __declspec ( dllexport ) 
#else
	#define NEURAL_NETWORK_LAYER_API __declspec ( dllimport ) 
#endif

// Include files.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// Class decleration.

using namespace std;

template class NEURAL_NETWORK_LAYER_API std::basic_string<char,std::char_traits<char>,std::allocator<char>>;
template class NEURAL_NETWORK_LAYER_API vector <double>;  
template class NEURAL_NETWORK_LAYER_API vector <vector <double>>;
                    
namespace common
{
	class NEURAL_NETWORK_LAYER_API NeuralNetworkLayer
	{
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Constants
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		const string C_EMPTY          = "";
		const string C_SPACE          = " ";
		const string C_NEW_LINE       = "\n";
		const string C_TAB_1          = "\t";
		const string C_TAB_2          = "\t\t";
		const string C_DOUBLE_QUOTE   = "\"";
		const char   C_PERIOD         = '.';
		const char   C_DIGIT_LEFT_PAD = ' ';
		const char   C_LABEL_LEFT_PAD = ' ';
		

		const string C_OUTPUT_UNIT_ID     = "id";
		const string C_OUTPUT_UNIT_BIAS   = "bias";
		const string C_OUTPUT_UNIT_WEIGHT = "w_";

		const int  C_DEFAULT_WEIGHT_PRECISION                    = 3;
		const int  C_DEFAULT_BIAS_PRECISION                      = 3;
		const int  C_DEFAULT_CONSOLE_DATA_PRECISION              = 3;
		const int  C_DEFAULT_CONSOLE_DATA_WIDTH                  = 7;
		const bool C_DEFAULT_CONSOLE_HEADER_ROW_ENABLED          = true;
		const bool C_DEFAULT_CONSOLE_ACTIVATION_FUNCTION_ENABLED = true;
		const int  C_DEFAULT_DSV_PRECISION                       = 0;
		const bool C_DEFAULT_DSV_HEADER_ROW_ENABLED              = true;

		const char*      C_BIN_FILE_HEADER_FILE_TYPE_NAME = "ANN Layer File.\n";
		static const int C_BIN_FILE_RESERVED_SPACE        = 38;

		const string C_MSG_EXCEPTION                         = "EXCEPTION: ";
		const string C_MSG_EXCEPTION_UNABLE_TO_OPEN_FILE_DSV = "Unable to open DSV file, ";
		const string C_MSG_EXCEPTION_UNABLE_TO_OPEN_FILE_TXT = "Unable to open text file, ";
		const string C_MSG_EXCEPTION_UNABLE_TO_OPEN_FILE_BIN = "Unable to open binary file, ";
		
	private:

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data type aliases
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		using ActivationFunctionType = enum
		{
			LINEAR,			// Direct linear weight function.			y = x
			STEP,			// Normalized step function.				y = step(x)
			SGN,			// Normalized sign function.				y = sgn(x)
			SIGMOID,		// Sigmoid logistics function.				y = sig(x)
			TANH,			// Hyperbolic tangent.						y = tanh(x)
			TANH2,			// Steep hyperbolic tangent.				y = tanh(2x)
			RELU,			// Rectified linear unit.					y = ReLU(x)
			SOFT_PLUS,		// SoftPlus function.						y = SoftPlus(x)
			CUBIC,			// Cubic polynomial (Similar to sin(x)).	y = x(3-x²)/2
			QUADRATIC,		// Peciprical quadratic function pair.		y = { x(x+2), x<0 }, { -x(x-2), x≥0 }
			PARABOLIC,		// Displaced parabolic function.			y = ((x+1)²)/2 - 1
			HYPERBOLIC,		// Hyperbolic function.						y = { 1/x, x≠0 }, { 0, x=0 } 
			PYTHAGORIAN		// Reciprical pythagorian functions.		y = { -sqrt(-x(x+2)), x<0 }, { sqrt(-x(x-2), , x≥0 )
		};

		using BinaryFileHeader = struct
		{
			char     file_type_name [ 16 ];						//  0 16: Set to "ANN Layer File.\n".
			char     zero_terminator;							// 16  1: Set to zero. e.g. 0x00.
			uint8_t  data_width;								// 17  1: 8, 16, 32, 64, 128, 255.
			uint32_t input_unit_count;							// 18  4: Number o finput units.
			uint32_t output_unit_count;							// 22  4: Number of output units.
			uint8_t  reserved [ C_BIN_FILE_RESERVED_SPACE ];	// 26  6: Reserved space.
		};

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		// Constructors and destructors

		NeuralNetworkLayer  ( void );
		NeuralNetworkLayer  ( int input_unit_count, int output_unit_count );
		NeuralNetworkLayer  ( const NeuralNetworkLayer& element );				// Copy constructor.
		~NeuralNetworkLayer ( void );											// Destructor.

		// Operators.

		NeuralNetworkLayer& operator = ( const NeuralNetworkLayer& element );	// Assignment operator.

		// Class managment functions.

		void initialize ( int input_unit_count, int output_unit_count );

		// Nerual network layer functions.

		void            set_weights					( double weight );
		void            randomize_weights			( double min, double max );
		void            gaussian_weights			( double min, double max );
		void            set_bias_vector				( double bias_term );
		void            randomize_bias_vector		( double min, double max );
		double          compute_activation_function	( double network_function );
		vector <double> compute_layer				( vector <double> x );
		

		// Analysis Functions.

		double unit_normal_sum ( int output_unit_index );
		double layer_normal_sum ();

		// File managment functions.

		void save_to_csv        ( string file_name );
		void save_to_tsv        ( string file_name );						        
		void save_to_dsv        ( string file_name, char delimiter );
		void save_to_dsv        ( string file_name, char delimiter, int precision );
		void save_to_dsv        ( string file_name, char delimiter, bool header_row_enabled );
		void save_to_dsv        ( string file_name, char delimiter, int precision, bool header_row_enabled );						        
		void save_to_dsv        ( ofstream& file, char delimiter );
		void save_to_dsv        ( ofstream& file, char delimiter, int precision );
		void save_to_dsv        ( ofstream& file, char delimiter, bool header_row_enabled );
		void save_to_dsv        ( ofstream& file, char delimiter, int precision, bool header_row_enabled );
						        
		void save_to_txt        ( string    file_name, int precision, bool header_row_enabled );
		void save_to_txt        ( ofstream& file, int precision, bool header_row_enabled );

		void save_to_bin_int8   ( string file_name );
		void save_to_bin_uint8  ( string file_name );
		void save_to_bin_double ( string file_name );

		// Display functions.

		string to_string       ();
		string to_string_int8  ();	// TODO:
		string to_string_uint8 ();	// TODO:
		string to_string       ( int precision, bool activation_function_enabled, bool header_row_enabled );

	private:

		string print_layer_header        ( char delimiter );
		string print_layer_header        ( int  width );
		string print_output_unit         ( long output_unit_index, int precision, char delimiter );
		string print_output_unit         ( long output_unit_index, int precision, int  width );
		string print_real_value          ( double value, int precision );
		string print_real_value          ( double value, int precision, int width );
		string print_activation_function ( ActivationFunctionType activation_function_type );

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Member data.
		//
		// Note:
		// - We are not using accessors and mutators in vafor of moden C++ best practices.
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		// Neural network layer properties.

		ActivationFunctionType   activation_function_type;	// Type of transfer function used to compute output units.
		int                      input_vector_size;			// Number of input units. i.e. Number of elements in input vector.
		int                      output_vector_size;		// Number of output units. i.e. Number of elements in output vector.    
		vector <vector <double>> weight_vectors;			// Input Weight Vectors [ Output Units ][ Input Units ].
		vector <double>          bias_vector;				// Bias vector.  
		vector <double>          output_vector;				// Updated after calling compute_layer.

	private:

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Accessors and mutators
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		// Accessors

		// ...

		// Mutators.

		// ...
	};
}


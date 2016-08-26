//--------------------------------------------------------------------------------------------------
// Project:	Common Artificial Inteligence Library
// Application: N/A
// Class:       NeuralNetwork
// Version:	1.0
// Date:	2014 (C++), ported from 2001 (Java), ported from 1994 (C)
// Author:	Rohin Gosling
//
// Description:
//
//   Neural network class.
//
//   Note:
//   - Ported from my Java version of this class.
//   - TODO: Check for C++ correctness.
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
// - Transfer Function
//   The transfer function defines the activation behaviour of a perceptron unit (neuron).
//
//--------------------------------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

using namespace std;

class NeuralNetworkLayer
{
	//--------------------------------------------------------------------------------------------------
	// Constants
	//--------------------------------------------------------------------------------------------------

public:

	const string C_EMPTY          = "";
        const string C_SPACE          = " ";
	const string C_NEW_LINE       = "\n";
        const string C_TAB_1          = "\t";
        const string C_TAB_2          = "\t\t";   
	const char   C_DIGIT_LEFT_PAD = ' ';   
	const char   C_LABEL_LEFT_PAD = ' ';   

	const string C_OUTPUT_UNIT_ID     = "id";
	const string C_OUTPUT_UNIT_BIAS   = "bias";
	const string C_OUTPUT_UNIT_WEIGHT = "w_";
		
	const int  C_DEFAULT_WEIGHT_PRECISION           = 3; 
	const int  C_DEFAULT_BIAS_PRECISION             = 3; 
        const int  C_DEFAULT_CONSOLE_DATA_PRECISION     = 3; 
	const int  C_DEFAULT_CONSOLE_DATA_WIDTH         = 7;
	const bool C_DEFAULT_CONSOLE_HEADER_ROW_ENABLED = true;
	const int  C_DEFAULT_DSV_PRECISION              = 0;
	const bool C_DEFAULT_DSV_HEADER_ROW_ENABLED     = true;

	const char*      C_BIN_FILE_HEADER_FILE_TYPE_NAME = "ANN Layer File.\n";
	static const int C_BIN_FILE_RESERVED_SPACE        = 38;

private:
	
	// ...

	//--------------------------------------------------------------------------------------------------
	// Data type aliases
	//--------------------------------------------------------------------------------------------------

public:

	using TransferFunctionType = enum
	{
		LINEAR,     // Direct weight function + bias.   y = x
                STEP,       // Step function.                   y = step(x)
                SGN,        // Sign function.                   y = sgn(x)
                SIGMOID,    // Sigmoid logistics function.      y = sig(x)
                TANH,       // Hyperbolic tangent.              y = tanh(x)
                TANH2       // Steep hyperbolic tangent.        y = tanh(2x)
	};

	using BinaryFileHeader = struct
	 {
		char     file_type_name[16];                            //  0 16: Set to "ANN Layer File.\n".
		char     zero_terminator;                               // 16  1: Set to zero. e.g. 0x00.
		uint8_t  data_width;	                                // 17  1: 8, 16, 32, 64, 128, 255.
		uint32_t input_unit_count;                              // 18  4: Number o finput units.
		uint32_t output_unit_count;                             // 22  4: Number of output units.
		uint8_t  reserved [ C_BIN_FILE_RESERVED_SPACE ];	// 26  6: Reserved space.
	};

	//--------------------------------------------------------------------------------------------------
	// Functions.
	//--------------------------------------------------------------------------------------------------

public:

	// Constructors and destructors

        NeuralNetworkLayer  ( void );
	NeuralNetworkLayer  ( int input_unit_count, int output_unit_count );
        ~NeuralNetworkLayer ( void );

	// Class managment functions.

	void initialize ( int input_unit_count, int output_unit_count );
	
        // Nerual network layer functions.

        void            set_weights           ( double weight );
	void            randomize_weights     ( double min, double max );
	void            gaussian_weights      ( double min, double max );
	void            setbias_vector        ( double bias_term );
	void            randomize_bias_vector ( double min, double max );
	vector <double> compute_layer         ( vector <double> x );

	// Analysis Functions.

	double unit_normal_sum  ( int output_unit_index );
	double layer_normal_sum ();

	// File managment functions.

	void save_to_csv ( string file_name );

	void save_to_tsv ( string file_name );	

	void save_to_dsv ( string file_name, char delimiter );	
	void save_to_dsv ( string file_name, char delimiter, int precision );
	void save_to_dsv ( string file_name, char delimiter, bool header_row_enabled );
	void save_to_dsv ( string file_name, char delimiter, int precision, bool header_row_enabled );

	void save_to_dsv ( ofstream& file, char delimiter );	
	void save_to_dsv ( ofstream& file, char delimiter, int precision );
	void save_to_dsv ( ofstream& file, char delimiter, bool header_row_enabled );
	void save_to_dsv ( ofstream& file, char delimiter, int precision, bool header_row_enabled );

	void save_to_txt ( string    file_name, int precision, bool header_row_enabled );
	void save_to_txt ( ofstream& file,      int precision, bool header_row_enabled );

	void save_to_bin_int8   ( string file_name );
	void save_to_bin_uint8  ( string file_name );
	void save_to_bin_double ( string file_name );

	// Display functions.

	string to_string       ();
	string to_string_int8  ();	// TODO:
	string to_string_uint8 ();	// TODO:
	string to_string       ( int precision, bool header_row_enabled );
		
private:

	string print_layer_header ( char delimiter );
	string print_layer_header ( int  width );
	string print_output_unit  ( long output_unit_index, int precision, char delimiter );
	string print_output_unit  ( long output_unit_index, int precision, int  width     );
	string print_real_value   ( double value, int precision );
	string print_real_value   ( double value, int precision, int width );
	

	//--------------------------------------------------------------------------------------------------
	// Member data.
	//
	// Note:
	// - We are not using accessors and mutators in vafor of moden C++ best practices.
	//--------------------------------------------------------------------------------------------------

public:

    // Neural network layer properties.

    TransferFunctionType     transfer_function_type;	// Type of transfer function used to compute output units.
    int                      input_vector_size;			// Number of input units. i.e. Number of elements in input vector.
    int                      output_vector_size;		// Number of output units. i.e. Number of elements in output vector.    
    vector <vector <double>> weight_vectors;			// Input Weight Vectors [ Output Units ][ Input Units ].
    vector <double>          bias_vector;				// Bias vector.  
	
private:

	// ...

    //--------------------------------------------------------------------------------------------------
	// Accessors and mutators
	//--------------------------------------------------------------------------------------------------

public:

	// Accessors

	// ...

	// Mutators.

	// ...
};


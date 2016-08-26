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

#include "stdafx.h"
#include "neural_network_layer.h"
#include "ai_math.h"

#include <stdio.h>

#include <sstream>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cstring>

using namespace std;

//--------------------------------------------------------------------------------------------------
// Constructor/s
//--------------------------------------------------------------------------------------------------

// Constructor 1/2

NeuralNetworkLayer::NeuralNetworkLayer ()
{
	initialize ( 0, 0 );
}

// Constructor 2/2

NeuralNetworkLayer::NeuralNetworkLayer ( int input_unit_count, int output_unit_count )
{
	initialize ( input_unit_count, output_unit_count );
}

//--------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------

NeuralNetworkLayer::~NeuralNetworkLayer ()
{
}

//--------------------------------------------------------------------------------------------------
// Initialize the network.
//--------------------------------------------------------------------------------------------------

void NeuralNetworkLayer::initialize ( int input_unit_count, int output_unit_count )
{
	// Declare local variables.

	double initial_weight_value { 0.0 };
	double initial_bias_value   { 0.0 };

	// Initialize this layer's weight matrix (Dendritic tree network). 
        
	this->weight_vectors.resize ( output_unit_count, vector <double> ( input_unit_count, initial_weight_value ));
	this->bias_vector.resize    ( output_unit_count, initial_bias_value );

	// Save the input/output unit counts.
        
	this->input_vector_size  = input_unit_count;
	this->output_vector_size = output_unit_count;
        
	// Initialize default layer parameters.
        
	this->transfer_function_type = TransferFunctionType::SIGMOID;
}

//--------------------------------------------------------------------------------------------------
// Set the Weights to a user specified value in the range, 0.0 and 1.0.
//--------------------------------------------------------------------------------------------------

void NeuralNetworkLayer::set_weights ( double weight )
{
	for ( int output_unit = 0; output_unit < this->output_vector_size; output_unit++ )
	{
		for ( int input_unit = 0; input_unit < this->input_vector_size; input_unit++ )
		{
			this->weight_vectors [ output_unit ][ input_unit ] = weight;
		}
	}
}

//--------------------------------------------------------------------------------------------------
// Set the Weights to a random value in the range, 0.0 and 1.0.
//--------------------------------------------------------------------------------------------------
    
void NeuralNetworkLayer::randomize_weights ( double min, double max )
{
	// Declare local variables.

	double distribution = 0.0;
	double value        = 0.0;

	// Initialize random number generator.

	random_device                      rd;
    mt19937                            random_generator    ( rd() );
    uniform_real_distribution <double> random_distribution ( 0.0, nextafter ( 1.0, DBL_MAX ) );

	// Initialize the weight matrix to random numbers.

	for ( int output_unit = 0; output_unit < this->output_vector_size; output_unit++ )
	{
		for ( int input_unit = 0; input_unit < this->input_vector_size; input_unit++ )
		{
			distribution = random_distribution ( random_generator );
			value = min + ( max - min ) * distribution;

			this->weight_vectors [ output_unit ][ input_unit ] = value;
		}
	}
}

//--------------------------------------------------------------------------------------------------
// Set the Weights to a Gaussian distributed value in the range, 0.0 and 1.0.
//--------------------------------------------------------------------------------------------------
    
void NeuralNetworkLayer::gaussian_weights ( double min, double max )
{	
	// Declare local variables.

	double distribution = 0.0;
	double value        = 0.0;

	// Initialize the weight matrix to random numbers.

	for ( int output_unit = 0; output_unit < this->output_vector_size; output_unit++ )
	{
		for ( int input_unit = 0; input_unit < this->input_vector_size; input_unit++ )
		{
			distribution = AIMath::gaussian_distribution();
			value        = min + ( max - min ) * distribution;

			this->weight_vectors [ output_unit ][ input_unit ] = value;
		}
	}
}

//--------------------------------------------------------------------------------------------------
// Set bias vector.
//--------------------------------------------------------------------------------------------------
    
void NeuralNetworkLayer::setbias_vector ( double bias_term )
{   
    for ( int output_unit = 0; output_unit < this->output_vector_size; output_unit++ )
    {
        this->bias_vector [ output_unit ] = bias_term;
    }
}

//--------------------------------------------------------------------------------------------------
// Randomize bias vector.
//--------------------------------------------------------------------------------------------------
    
void NeuralNetworkLayer::randomize_bias_vector ( double min, double max )
{   
	// Declare local variables.

	double distribution = 0.0;
	double value        = 0.0;

	// Initialize random number generator.

	random_device                      rd;
    mt19937                            random_generator    ( rd() );
    uniform_real_distribution <double> random_distribution ( 0.0, nextafter ( 1.0, DBL_MAX ) );

	// Initialize bias vector.

    for ( int output_unit = 0; output_unit < this->output_vector_size; output_unit++ )
    {
		distribution = random_distribution ( random_generator );
		value        = min + ( max - min ) * distribution;

        this->bias_vector [ output_unit ] = value;
    }
}

//--------------------------------------------------------------------------------------------------
// Compute output vector for the layer.
//--------------------------------------------------------------------------------------------------
    
vector <double> NeuralNetworkLayer::compute_layer ( vector <double> x )
{
	// Declare and initialize local variables.

	vector <double> y ( this->output_vector_size );								// Output vector.    
    vector <double> weight_function;											// Weight function vector result.
    double          network_function  = 0.0;									// Network function result.
    double          transfer_function = 0.0;									// Transfer function result.
        
	// Compute the netwrok layer.

    for ( int output_unit_index = 0; output_unit_index < this->output_vector_size; output_unit_index++ )
    {
        // Retrieve input vector and weight vector.
            
        vector <double> w = this->weight_vectors [ output_unit_index ];	// Input weight vector.
            
        // Evaluate weight function. 
                        
        weight_function = AIMath::hadamard_product ( w, x );					// Apply relationship between the input vector and the weight vector.
            
        // Evaluate network function.
                        
        network_function = AIMath::vector_sum ( weight_function );				// Sum up the elements of the weight function vector.
        network_function += this->bias_vector [ output_unit_index ];			// Add the bias term.
            
        // Evaluate transfer function.
            
        switch ( this->transfer_function_type )
        {
            case LINEAR:    transfer_function = network_function;                           break;
			case SGN:       transfer_function = AIMath::sgn     ( network_function       );  break;
			case STEP:      transfer_function = AIMath::step    ( network_function       );  break;
			case SIGMOID:   transfer_function = AIMath::sigmoid ( network_function       );  break;                
			case TANH:      transfer_function = tanh            ( network_function       );  break;                
			case TANH2:     transfer_function = tanh            ( network_function * 2.0 );  break;
        }
            
        y [ output_unit_index ] = transfer_function;            
    }

	// REturn the output vector.
        
    return y;
}

//--------------------------------------------------------------------------------------------------
// output_unit_sum
//--------------------------------------------------------------------------------------------------

double NeuralNetworkLayer::unit_normal_sum ( int output_unit_index )
{
	// Declare an initialize local variables.

	double normal_sum       { 0.0 };
	double sum              { 0.0 };	
	double weight           { 0.0 };
	double bias             { 0.0 };
	int    input_unit_count { this->input_vector_size };

	// Retrieve bias value for this output unit.
		
	bias = this->bias_vector [ output_unit_index ];

	// Compute sum of weights + bias.

	sum  = bias;

	for ( int input_unit_index = 0; input_unit_index < input_unit_count; input_unit_index++ )
	{
		// Retrieve output unit weight data.
		
		weight = this->weight_vectors [ output_unit_index ][ input_unit_index ];

		// Compute sum.

		sum += weight;
	}

	// Normalize sum.

	normal_sum = sum / input_unit_count;

	// Return result.

	return normal_sum;
}

//--------------------------------------------------------------------------------------------------
// layer_sum
//--------------------------------------------------------------------------------------------------

double NeuralNetworkLayer::layer_normal_sum ()
{
	// Declare and initialize local variables.

	double normal_sum        { 0.0 };
	double sum               { 0.0 };
	int    output_unit_count { this->output_vector_size };

	// Compute sum of layer normal sums.

	sum = 0.0;

	for ( int output_unit_index = 0; output_unit_index < output_unit_count; output_unit_index++ )
	{
		sum += unit_normal_sum ( output_unit_index );
	}

	// Normalise sum.

	normal_sum = sum / output_unit_count;

	// Return result;

	return normal_sum;
}

//--------------------------------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------------------------------

void NeuralNetworkLayer::save_to_csv ( string file_name )
{
	save_to_dsv ( file_name, ',' );
}

//--------------------------------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------------------------------

void NeuralNetworkLayer::save_to_tsv ( string file_name )
{
	save_to_dsv ( file_name, '\t' );
}

//--------------------------------------------------------------------------------------------------
// save_to_dsv
//--------------------------------------------------------------------------------------------------

// Overload 1/4

void NeuralNetworkLayer::save_to_dsv ( string file_name, char delimiter )
{
	save_to_dsv ( file_name, delimiter, C_DEFAULT_DSV_PRECISION, C_DEFAULT_DSV_HEADER_ROW_ENABLED );
}

// Overload 2/4

void NeuralNetworkLayer::save_to_dsv ( string file_name, char delimiter, int precision )
{
	save_to_dsv ( file_name, delimiter, precision, C_DEFAULT_DSV_HEADER_ROW_ENABLED );
}

// Overload 3/4

void NeuralNetworkLayer::save_to_dsv ( string file_name, char delimiter, bool header_row_enabled )
{
	save_to_dsv ( file_name, delimiter, C_DEFAULT_DSV_PRECISION, header_row_enabled );
}

// Overload 4/4

void NeuralNetworkLayer::save_to_dsv ( string file_name, char delimiter, int precision, bool header_row_enabled )
{
	// Open file for appending.

	ofstream file;
	file.open ( file_name );

	// Write header row, if enabled.

	save_to_dsv ( file, delimiter, precision, header_row_enabled );

	// Close file.

	file.close ();
}

// Overload 5/8

void NeuralNetworkLayer::save_to_dsv ( ofstream& file, char delimiter )
{
	save_to_dsv ( file, delimiter, C_DEFAULT_DSV_PRECISION, C_DEFAULT_DSV_HEADER_ROW_ENABLED );
}

// Overload 6/8

void NeuralNetworkLayer::save_to_dsv ( ofstream& file, char delimiter, int precision )
{
	save_to_dsv ( file, delimiter, precision, C_DEFAULT_DSV_HEADER_ROW_ENABLED );
}

// Overload 7/8

void NeuralNetworkLayer::save_to_dsv ( ofstream& file, char delimiter, bool header_row_enabled )
{
		save_to_dsv ( file, delimiter, C_DEFAULT_DSV_PRECISION, header_row_enabled );
}

// Overload 8/8

void NeuralNetworkLayer::save_to_dsv ( ofstream& file, char delimiter, int precision, bool header_row_enabled )
{
	// Initialize local variables.

	int output_unit_count { this->output_vector_size };		// Retrieve the number of output units.	
                
	// Print layer data header.

	if ( header_row_enabled ) file << print_layer_header ( delimiter );

	// Print layer data.

	for ( int output_unit_index = 0; output_unit_index < output_unit_count; output_unit_index++ )
	{
		file << print_output_unit ( output_unit_index, precision, delimiter );
	}   
}

//--------------------------------------------------------------------------------------------------
// save_to_fixed_width
//--------------------------------------------------------------------------------------------------

// Overload 1/2

void NeuralNetworkLayer::save_to_txt ( string file_name, int precision, bool header_row_enabled )
{
	// Open file for appending.

	ofstream file;
	file.open ( file_name );

	// Write header row, if enabled.

	save_to_txt ( file, precision, header_row_enabled );

	// Close file.

	file.close ();
}

// Overload 2/2

void NeuralNetworkLayer::save_to_txt ( ofstream& file, int precision, bool header_row_enabled )
{
	file << to_string ( precision, header_row_enabled );
}

//--------------------------------------------------------------------------------------------------
// save_to_bin
//--------------------------------------------------------------------------------------------------

// Overload 1/3

void NeuralNetworkLayer::save_to_bin_int8 ( string file_name )
{

}

// Overload 2/3

void NeuralNetworkLayer::save_to_bin_uint8 ( string file_name )
{
#if 0
	// Configure binary file header

	BinaryFileHeader file_header =
	{
		(char)     "",
		(char)     0x00,
		(uint8_t)  1,
		(uint16_t) this->input_vector_size,
		(uint16_t) this->output_vector_size,
		(uint8_t)  string ( 0xFF, C_BIN_FILE_RESERVED_SPACE ).c_str()
	};
	strcpy ( file_header.file_type_name, C_BIN_FILE_HEADER_FILE_TYPE_NAME );

	// Open the file.

	ofstream file;
	file = ofstream ( file_name, ios::binary );

	// Write the header.

	int header_size = sizeof ( file_header );

	file.write ( (char*) &file_header, header_size );

	// Write layer data.

	// Close file.

	file.close ();
#endif
}

// Overload 3/3

void NeuralNetworkLayer::save_to_bin_double ( string file_name )
{
}


//--------------------------------------------------------------------------------------------------
// to_string
//--------------------------------------------------------------------------------------------------

// Overload 1/2.

string NeuralNetworkLayer::to_string ()
{       
	return to_string ( C_DEFAULT_CONSOLE_DATA_PRECISION, C_DEFAULT_CONSOLE_HEADER_ROW_ENABLED );
}

// Overload 2/2.

string NeuralNetworkLayer::to_string ( int precision, bool header_row_enabled )
{
	// Initialize local variables.

	string s;
	int    output_unit_count { this->output_vector_size };		// Retrieve the number of output units.
	int    width             { precision + 4 };					// We add 4, for the decimal point, the whole number part, the sign, and one leading space.
                
	// Print layer data header.

	if ( header_row_enabled ) s += print_layer_header ( width );

	// Print layer data.

	for ( int output_unit_index = 0; output_unit_index < output_unit_count; output_unit_index++ )
	{
		s += print_output_unit ( output_unit_index, precision, width );
	}
        
	// Return output string.
        
	return s;
}

//--------------------------------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------------------------------

// Overload 1/2

string NeuralNetworkLayer::print_layer_header ( char delimiter )
{	
	// Declare and intialize local variables.

	stringstream ss;
	int          input_unit_count { this->input_vector_size };

	// Output output unit ID header, and bias header.

	ss << C_OUTPUT_UNIT_ID  << delimiter;
	ss << C_OUTPUT_UNIT_BIAS;

	// Include a delimiter afeter the bias, only if there are weights for the unit.

	if ( input_unit_count > 0 )
	{
		ss << delimiter;
	}

	for ( int input_unit_index = 0; input_unit_index < input_unit_count; input_unit_index++ )
	{
		// Append the next weight header.

		ss << C_OUTPUT_UNIT_WEIGHT << input_unit_index;

		// Append a delimiter, unless we are at the last element.

		if ( input_unit_index < input_unit_count - 1 )
		{
			ss << delimiter;
		}		
	}

	// Append a new line.

	ss << endl;

	// Return string.

	return ss.str ();
}

// Overload 2/2

string NeuralNetworkLayer::print_layer_header ( int width )
{	
	// Declare and intialize local variables.

	stringstream ss;
	string       weight_string;
	int          input_unit_count { this->input_vector_size };

	// Compile row string.

	ss << setfill ( C_LABEL_LEFT_PAD ) << setw ( width ) << C_OUTPUT_UNIT_ID;
	ss << setfill ( C_LABEL_LEFT_PAD ) << setw ( width ) << C_OUTPUT_UNIT_BIAS;

	for ( int input_unit_index = 0; input_unit_index < input_unit_count; input_unit_index++ )
	{	
		// Compile weight string.

		weight_string =  C_OUTPUT_UNIT_WEIGHT + std::to_string ( input_unit_index );		

		// Append the next weight.

		ss << setfill ( C_LABEL_LEFT_PAD ) << setw ( width ) << weight_string;
	}

	// Append an endo fo line.

	ss << endl;

	// Return string.

	return ss.str ();
}

//--------------------------------------------------------------------------------------------------
// print_output_unit
//--------------------------------------------------------------------------------------------------

// Overload 1/2

string NeuralNetworkLayer::print_output_unit ( long output_unit_index, int precision, char delimiter )
{
	// constants.

	const int WIDTH = 0;

	// Declare and intialize local variables.

	stringstream ss;
	int          input_unit_count { this->input_vector_size };
	double       bias             { this->bias_vector [ output_unit_index ] };
	double       weight           { 0.0 };

		// Clip precision.

	precision = ( precision < 0 ) ? 0 : precision;

	// Output the slected unit ID.

	ss << output_unit_index << delimiter;

	// Output the bias value for the selceted unit.

	ss << print_real_value ( bias, precision, WIDTH );

	// Include a delimiter afeter the bias, only if there are weights for the unit.

	if ( input_unit_count > 0 )
	{
		ss << delimiter;
	}	

	// Output weights.

	for ( int input_unit_index = 0; input_unit_index < input_unit_count; input_unit_index++ )
	{
		// Retrieve input weight.
				
		weight = this->weight_vectors [ output_unit_index ][ input_unit_index ];

		// Append next weight.

		ss << print_real_value ( weight, precision, WIDTH );

		// Append a delimiter, unless we are at the last element.

		if ( input_unit_index < input_unit_count - 1 )
		{
			ss << delimiter;
		}
	}

	// Append a new line.

	ss << endl;

	// Return string.

	return ss.str ();
}

// Overload 2/2

string NeuralNetworkLayer::print_output_unit ( long output_unit_index, int precision, int width )
{
	// Declare and intialize local variables.

	stringstream ss;
	int          input_unit_count { this->input_vector_size };
	double       bias             { this->bias_vector [ output_unit_index ] };
	double       weight           { 0.0 };

	// Clip precision.

	precision = ( precision < 0 ) ? 0 : precision;

	// Output the slected unit ID.

	ss << setfill ( C_DIGIT_LEFT_PAD ) << setw ( width ) << output_unit_index;

	// Output the bias value for the selceted unit.

	ss << print_real_value ( bias, precision, width );
	
	// Output weights.

	for ( int input_unit_index = 0; input_unit_index < input_unit_count; input_unit_index++ )
	{
		// Retrieve input weight.
				
		weight = this->weight_vectors [ output_unit_index ][ input_unit_index ];

		// Append next weight.

		ss << print_real_value ( weight, precision, width );
	}

	// Append a new line.

	ss << endl;

	// Return string.

	return ss.str ();
}


//--------------------------------------------------------------------------------------------------
// print_real_value
//--------------------------------------------------------------------------------------------------

string NeuralNetworkLayer::print_real_value ( double value, int precision, int width )
{
	// Declare and initialize local variables.

	stringstream ss;

	// clip width;

	width = ( width < 0 ) ? 0 : width;

	// Format value.

	if ( precision > 0 )
	{
		if ( width > 0 )
		{
			ss << setfill ( C_DIGIT_LEFT_PAD ) << setw ( width ) << setprecision ( precision ) << fixed << value;
		}
		else
		{
			ss << setprecision ( precision ) << fixed << value;
		}
	}
	else
	{
		if ( width > 0 )
		{
			ss << setfill ( C_DIGIT_LEFT_PAD ) << setw ( width ) << value;
		}
		else
		{
			ss << value;
		}
	}

	// Retrun string.

	return ss.str ();
}







































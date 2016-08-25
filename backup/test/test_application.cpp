//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Project:	    Entelect 100k Challenge.
// Application: Neural Network test application.
// Class:	    TestApplication
// Version:	    1.0
// Date:	    2015
// Author:	    Rohin Gosling
//
// Description:
//
//   Test class, used to test the NeuralNetwork class.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "test_application.h"
#include "static_utility.h"
#include "neural_network_layer.h"
#include "neural_network.h"

#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;	// We will be using some timming functions for measurement.
using namespace common;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Constructor 1/1
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TestApplication::TestApplication ( int argc, char* argv [] ) : ConsoleApplication { argc, argv }
{
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TestApplication::~TestApplication ()
{
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - ExecuteCommandInterpreter
//
// DESCRIPTION:
//
// - Execute commands passed in through the command line.
//
// - ExecuteCommandInterpreter should be overloaded in a derived class, and not
//   called directly.
//
//   1. Extend ConsoleApplication.
//   2. Overload ExecuteCommandInterpreter() to implement commands.
//   3. Call an instance of the derived class in Main.cpp.
//
// ARGUMENTS:
//
// - N/A
//
// RETURN VALUE:
//
// - N/A.
//
// PRE-CONDITIONS:
//
// - N/A
//   
// POST-CONDITIONS:
//
// - The apropriate commands, as defined in a derived class, have been executed.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_interpreter ( vector <string> command_line ) 
{
	// Declare local variables.

	int test_case = 5;

	// Get the number of arguments on the command line, including the command it's self.

	int argument_count { (int) command_line.size () };
	int argument_index { 0 };

	// Execute commands.

	switch ( argument_count )
	{
		case 1:
			switch ( test_case )
			{
				case 0: command_main_program ();	break;
				case 1: command_test_1 ();			break;		// Test 2D vector element ordering for weight amtrix.
				case 2: command_test_2 ();			break;		// Test Layer.
				case 3: command_test_3 ();			break;
				case 4: command_test_4 ();			break;		// Layer level boolean function test. OR and AND.
				case 5: command_test_5 ();			break;		// NeuralNetwork test 1.
			}
			break;

		case 2:			
			break;

		default:			
			break;
	}
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_main_program ()
{
	print_program_info ( C_PROGRAM_INFO_NA, C_PROGRAM_INFO_NA, "Main program" );
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - command_test_1
//
// DESCRIPTION:
//
// - Test 2D array managment code.
// - We are not testing any of the actual NeuralNetwork or NeuralNetworkLayer functions here.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_test_1 ()
{
	// Declare local variables.

	vector <vector <int>> layer;
	int input_unit_count  = 8;
	int output_unit_count = 4;
	int input_unit_index  = 0;
	int initial_value     = 0;

	// Initialize network layer.

	layer.resize ( output_unit_count, vector<int> ( input_unit_count, initial_value ) );

	// Print network.
	
	cout << "\nInputs:\t\t";	
	for ( int i=0; i < input_unit_count; i++ ) cout << i << "\t";
	cout << "\n\t\t";
	for ( int i=0; i < input_unit_count; i++ ) cout << "-" << "\t";
	cout << "\n\n\n";
	
	input_unit_index = 0;
	for ( vector<int> v : layer )
	{
		cout << "Output " << input_unit_index << "|\t";		
		for ( int i : v ) cout << i << "\t";
	    cout << "\n\n";
		++input_unit_index;
	}
	cout << "\n";

	// Set some values.

	layer [ 1 ][ 2 ] = 100;

	// Print network.
	
	cout << "\nInputs:\t\t";	
	for ( int i=0; i < input_unit_count; i++ ) cout << i << "\t";
	cout << "\n\t\t";
	for ( int i=0; i < input_unit_count; i++ ) cout << "-" << "\t";
	cout << "\n\n\n";
	
	input_unit_index = 0;
	for ( vector<int> v : layer )
	{
		cout << "Output " << input_unit_index << "|\t";		
		for ( int i : v ) cout << i << "\t";
	    cout << "\n\n";
		++input_unit_index;
	}
	cout << "\n";
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - command_test_2
//
// DESCRIPTION:
//
// - Testing analysis functions in the NeuralNetworkLayer class.
// - Test file output functions.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_test_2 ()
{
	// Declare local variables.

	NeuralNetworkLayer ann_layer { 8, 8 };

	string file_path   { "R:\\Projects\\common\\machine_learning\\cpp\\neural_network\\Debug\\Output\\" };
	string file_name_0 { "test_file_0.dsv"     };
	string file_name_1 { "test_file_1.csv"     };
	string file_name_2 { "test_file_2.tsv"     };
	string file_name_3 { "test_file_3.txt"     };
	string file_name_4 { "test_file_uint8.bin" };
	string s;
	int    precision                   { 3 };	
	bool   header_row_enabled          { true };
	bool   activation_function_enabled { true };
	char   delimiter                   { ',' };

	// Initialize layer parameters.

	ann_layer.set_bias_vector   ( 0.0 );		
	ann_layer.randomize_weights ( 0.0, 1.0 );

	s += ann_layer.to_string ( precision, header_row_enabled, activation_function_enabled );
	
	cout << s;

	// Test analsys functions.

	cout << endl;

	for ( int normal_index = 0; normal_index < ann_layer.output_vector_size; normal_index++ )
	{
		cout << "Sum ( Unit Normal [" << normal_index << "] ) = " << ann_layer.unit_normal_sum ( normal_index ) << endl;
	}

	cout << endl << "Sum ( units ) = " << ann_layer.layer_normal_sum () << endl;

	cout << endl << "Output Vector Size = " << ann_layer.output_vector.size() << endl;

	// Test file saving.

	ann_layer.save_to_dsv       ( file_path + file_name_0, delimiter, precision, header_row_enabled );
	ann_layer.save_to_csv       ( file_path + file_name_1 );
	ann_layer.save_to_tsv       ( file_path + file_name_2 );
	ann_layer.save_to_txt       ( file_path + file_name_3, 4, true );
	ann_layer.save_to_bin_uint8 ( file_path + file_name_4 );
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - command_test_3
//
// DESCRIPTION:
//
// - Test the compute_layer function.
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_test_3 ()
{
	// Display test info.

	cout << endl << "TEST FUNCTION: command_test_3" << endl << endl;

	// Declare local variables.
	
	string s;
	int    precision                   { 5 };	
	bool   header_row_enabled          { true };
	bool   activation_function_enabled { true };
	char   delimiter                   { ',' };

	// Initialize layer.

	int    input_unit_count  { 3 };
	int    output_unit_count { 2 };

	NeuralNetworkLayer ann_layer { input_unit_count, output_unit_count };

	ann_layer.activation_function_type = NeuralNetworkLayer::ActivationFunctionType::TANH;
	ann_layer.set_bias_vector   (  0.0 );			
	//ann_layer.set_weights     (  0.0 );
	ann_layer.randomize_weights ( -0.5, 0.5 );

	// Display layer data to console.

	s += ann_layer.to_string ( precision, header_row_enabled, activation_function_enabled );	
	cout << s << endl;

	// Initialize layer input and output.

	vector <double> x { -1.0, 0.0, 1.0 };
	vector <double> y ( output_unit_count );

	// Test layer computation.

	y = ann_layer.compute_layer ( x );

	// Display input and output.

	print_vector ( x, precision, "Input Vector:",  "x" );
	print_vector ( y, precision, "Output Vector:", "y" );
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - command_test_4
//
// DESCRIPTION:
//
// - Layer level boolean function test. 
//   y = OR  ( x0, x1 )
//   y = AND ( x0, x1 )
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::command_test_4 ()
{
		// Display test info.

	cout << endl;
	cout << "TEST FUNCTION: command_test_4" << endl;
	cout << "DESCRIPTION:   Boolean function test." << endl;
	cout << endl;
	cout << "Perceptron:" << endl;
	cout << endl;

	// Declare local variables.
	
	string s;
	int    precision                   { 3 };	
	bool   header_row_enabled          { true };
	bool   activation_function_enabled { true };
	char   delimiter                   { ',' };

	// Initialize number of input and output units.

	int    input_unit_count  { 2 };
	int    output_unit_count { 1 };

	NeuralNetworkLayer ann_layer { input_unit_count, output_unit_count };

	ann_layer.activation_function_type = NeuralNetworkLayer::ActivationFunctionType::QUADRATIC;				
	ann_layer.set_bias_vector ( 0.0 );
	ann_layer.set_weights     ( 0.0 );	

	// Manualy program weight vector/s.
	//
	// - Perceptron weights and bias to implement an AND gate.
	//   w0 =  0.5 
	//   w1 =  0.5
	//   b  = -0.8
	//
	// - Perceptron weights and bias to implement an OR gate. 
	//   w0 =  0.5 
	//   w1 =  0.5
	//   b  = -0.3

	vector<double> weight_vector { 0.5, 0.5 };
	int output_unit_index { 0 };

	ann_layer.weight_vectors [ output_unit_index ] = weight_vector;	
	ann_layer.set_bias_vector ( -0.3 );
	

	// Display layer data to console.

	s += ann_layer.to_string ( precision, header_row_enabled, activation_function_enabled );	
	cout << s << endl;

	// Initialize layer input and output.

	vector <double> x { 1.0, 0.0 };
	vector <double> y { 0.0 };

	// Test layer computation.

	y = ann_layer.compute_layer ( x );

	// Display input and output.

	print_vector ( x, precision, "Input Vector:",  "x" );
	print_vector ( y, precision, "Output Vector:", "y" );
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// FUNCTION:
//
// - command_test_4
//
// DESCRIPTION:
//
// - Layer level boolean function test. 
//   y = OR  ( x0, x1 )
//   y = AND ( x0, x1 )
//
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#define TIME_START time_point_cast <milliseconds> (time_start).time_since_epoch().count()
#define TIME_STOP  time_point_cast <milliseconds> (steady_clock::now ()).time_since_epoch().count()

void TestApplication::command_test_5 ()
{
	// Define local constants.

	const string LOG_SYSTEM = "SYSTEM: ";
	const string LOG_TAB    = " ";

	// Initialize clock.

	steady_clock::time_point time_start = steady_clock::now ();	

	// Display test info.

	cout << endl;
	cout << "TEST FUNCTION: command_test_5" << endl;
	cout << "DESCRIPTION:   NeuralNetwork Class Test 1.0." << endl;
	cout << endl;

	// Declare local display variables.
	
	string s;
	int    precision                   { 3 };	
	bool   header_row_enabled          { true };
	bool   activation_function_enabled { true };

	// Initialize neural network.

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Building neural netowrk." ) << endl;

		vector<int>   network_architecture { 2, 2, 1 };
		NeuralNetwork ann                  { network_architecture };

		cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Neural netowrk ready." )        << endl;
		cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Initializing neural netowrk." ) << endl;

		if ( 0 ) ann.set_bias_vectors       (  0.0 );
		if ( 0 ) ann.set_weights            (  0.0 );
		if ( 0 ) ann.randomize_weights      ( -0.5, 0.5 );
		if ( 0 ) ann.randomize_bias_vectors ( -0.5, 0.5 );
		if ( 1 ) ann.neural_network [0].activation_function_type = NeuralNetworkLayer::ActivationFunctionType::STEP;
		if ( 1 ) ann.neural_network [1].activation_function_type = NeuralNetworkLayer::ActivationFunctionType::STEP;

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Neural netowrk initialized." ) << endl;

	// Train neural netowrk

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Training neural netowrk." ) << endl;

		vector<vector<vector<double>>> weight { { { 1.0, 1.0 }, { 1.0, 1.0 } }, { { 1.0,  -1.0 } } };
		vector<vector<double>>         bias   { {      0.5,          1.5     }, {      0.5       } };

		ann.neural_network [ 0 ].weight_vectors = weight [ 0 ];
		ann.neural_network [ 1 ].weight_vectors = weight [ 1 ];
		ann.neural_network [ 0 ].bias_vector    = bias   [ 0 ];
		ann.neural_network [ 1 ].bias_vector    = bias   [ 1 ];

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Neural netowrk trained." ) << endl;

	// Initialize input and output vectors.

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Initializing input and output vectors." ) << endl;

		vector<double> x { 0, 0 };
		vector<double> y { 0 };

		if (0) for ( int i=0; i < network_architecture [ 0 ]; i++ )	x.push_back ( 0.0 );
		if (0) for ( int i=0; i < network_architecture [ 1 ]; i++ )	y.push_back ( 0.0 );		

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Input and output vectors initialized." ) << endl;

	// Test neural network.

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Computing network (Feed forward execution)." ) << endl;

		y = ann.compute_network ( x );

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Network computation complete." ) << endl;

	// Display layer data to console.

#if 1

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Compiling neural network display string." ) << endl;	

		s += ann.to_string ( precision, activation_function_enabled, header_row_enabled );	

	cout << log_to_string ( TIME_START, TIME_STOP, LOG_SYSTEM + "Display string ready." ) << endl;	

	// Display input and output.

	cout << endl;
	print_vector ( x, precision, "Input Vector:",  "x" );
	print_vector ( y, precision, "Output Vector:", "y" );

	// Display neural network data.

	cout << endl << "Neural Network:" << endl << endl;
	cout << "\ts.size() = " << s.size()/1048576.0 << " MB" << endl;
	cout << "\ts.size() = " << s.size()/1024.0    << " KB" << endl;
	cout << "\ts.size() = " << s.size()           << " B" << endl;
	cout << endl;
	cout << s << endl;

#endif
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//print_program_info
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Overload 1/2

void TestApplication::print_program_info ()
{
	print_program_info ( C_PROGRAM_INFO_NA, C_PROGRAM_INFO_NA, C_PROGRAM_INFO_NA );
}

// Overload 2/2

void TestApplication::print_program_info ( string class_name, string function_name, string description )	
{
	cout << endl;

	cout << C_PROGRAM_INFO_PROJECT  << endl;
	cout << C_PROGRAM_INFO_PROGRAM  << endl;
	cout << C_PROGRAM_INFO_VERSION  << endl;
	cout << C_PROGRAM_INFO_DATE     << endl;
	cout << C_PROGRAM_INFO_AUTHOR   << endl;

	if ( class_name.length ()    > 0 ) cout << C_PROGRAM_INFO_CLASS       << class_name    << endl;
	if ( function_name.length () > 0 ) cout << C_PROGRAM_INFO_FUNCTION    << function_name << endl;
	if ( description.length ()   > 0 ) cout << C_PROGRAM_INFO_DESCRIPTION << description   << endl;
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// print_vector
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

void TestApplication::print_vector ( vector <double> v, int precision, string label, string vector_name )
{
	// Define lambdas

	auto new_line = [] () { cout << endl; };

	// Initialize local variables.

	string element_value_string;
	int    width { precision + 3 };		// We add 3, for the sign, one decimal unit, and a decimal point.

	// Print the vector label.

	cout << label << endl;
	new_line ();

	// Print out the vector data to the console.

	for ( int index = 0; index < (signed) v.size(); index++ )
	{
		element_value_string = StaticUtility::to_aligned_string ( v [ index ], precision, width, StaticUtility::ALIGN_RIGHT );
		cout << setw ( 5 ) << setprecision ( 5 ) << fixed;
		cout << vector_name << "[" << index << "] = " << element_value_string << endl;			
	} 
	new_line ();
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// log
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

string TestApplication::log_to_string ( long long time_point, long long time_stop, string message )
{
	// Compute elapsed time.

	long long elappsed_time = time_stop - time_point;

	// Compile log string.
	
	string s_time { StaticUtility::time_to_string ( elappsed_time ) };
	string s_log  { s_time + " " + message };

	// Return string to caller.

	return s_log;
}




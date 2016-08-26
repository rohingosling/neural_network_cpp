//--------------------------------------------------------------------------------------------------
// Project:	Entelect 100k Challenge.
// Application: Neural Network test application.
// Class:	TestApplication
// Version:	1.0
// Date:	2015
// Author:	Rohin Gosling
//
// Description:
//
//   Test class, used to test the NeuralNetwork class.
//
//--------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "test_application.h"
#include "neural_network_layer.h"

#include <iostream>
#include <string>

using namespace std;

//--------------------------------------------------------------------------------------------------
// Constructor 1/1
//--------------------------------------------------------------------------------------------------

TestApplication::TestApplication ( int argc, char* argv [] ) : ConsoleApplication { argc, argv }
{
}

//--------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------

TestApplication::~TestApplication ()
{
}

//--------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------

void TestApplication::command_interpreter ( vector <string> command_line ) 
{
	// Declare local variables.

	int test_case = 2;

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
			}
			break;

		case 2:			
			break;

		default:			
			break;
	}
}

//--------------------------------------------------------------------------------------------------

void TestApplication::command_main_program ()
{
	print_program_info ( C_PROGRAM_INFO_NA, C_PROGRAM_INFO_NA, "Main program" );
}

//--------------------------------------------------------------------------------------------------

void TestApplication::command_test_1 ()
{
	// Declare local variables.

	vector <vector <int>> layer;
	int input_unit_count  = 16;
	int output_unit_count = 8;
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

	layer [ 5 ][ 1 ] = 100;

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

//--------------------------------------------------------------------------------------------------

void TestApplication::command_test_2 ()
{
	// Declare local variables.

	NeuralNetworkLayer ann_layer { 16, 32 };

	string file_name_0 { "R:\\Projects\\100k Challange\\2015\\Source\\Tools\\neural_network\\Debug\\test_file_0.dsv"     };
	string file_name_1 { "R:\\Projects\\100k Challange\\2015\\Source\\Tools\\neural_network\\Debug\\test_file_1.csv"     };
	string file_name_2 { "R:\\Projects\\100k Challange\\2015\\Source\\Tools\\neural_network\\Debug\\test_file_2.tsv"     };
	string file_name_3 { "R:\\Projects\\100k Challange\\2015\\Source\\Tools\\neural_network\\Debug\\test_file_3.txt"     };
	string file_name_4 { "R:\\Projects\\100k Challange\\2015\\Source\\Tools\\neural_network\\Debug\\test_file_uint8.bin" };
	string s;
	int    precision         { 3 };	
	bool   enable_header     { true };
	char   delimiter         { ',' };

	// Test console functions.

	ann_layer.setbias_vector    ( 0.0 );		
	ann_layer.randomize_weights ( 0.0, 1.0 );

	s += ann_layer.to_string ( precision, enable_header );
	
	cout << s;

	// Test analsys functions.

	cout << endl;

	for ( int normal_index = 0; normal_index < ann_layer.output_vector_size; normal_index++ )
	{
		cout << "Unit normal = " << ann_layer.unit_normal_sum ( normal_index ) << endl;
	}

	cout << endl << "Layer normal = " << ann_layer.layer_normal_sum () << endl;

	// Test file saving.

	ann_layer.save_to_dsv       ( file_name_0, delimiter, precision, enable_header );
	ann_layer.save_to_csv       ( file_name_1 );
	ann_layer.save_to_tsv       ( file_name_2 );
	ann_layer.save_to_txt       ( file_name_3, 4, true );
	ann_layer.save_to_bin_uint8 ( file_name_4 );

}

//--------------------------------------------------------------------------------------------------

void TestApplication::command_test_3 ()
{
}

//--------------------------------------------------------------------------------------------------

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

//--------------------------------------------------------------------------------------------------


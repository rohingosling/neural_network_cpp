//--------------------------------------------------------------------------------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------------------------------------------------------------------------------

#pragma once

#include "console_application.h"

using namespace std;
using namespace common;

class TestApplication :	public ConsoleApplication
{
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Constants
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

	// Program information.

	const string C_PROGRAM_INFO_PROJECT     = "PROJECT     : Simulated Annealing - ANN Trainer.";
	const string C_PROGRAM_INFO_PROGRAM     = "PROGRAM     : Neural Network Test program.";
	const string C_PROGRAM_INFO_VERSION     = "VERSION     : 1.0";
	const string C_PROGRAM_INFO_DATE        = "DATE        : 2015-08-12";
	const string C_PROGRAM_INFO_AUTHOR      = "AUTHOR      : Rohin Gosling";
	const string C_PROGRAM_INFO_CLASS       = "CLASS       : ";	
	const string C_PROGRAM_INFO_FUNCTION    = "FUNCTION    : ";	
	const string C_PROGRAM_INFO_DESCRIPTION = "DESCRIPTION : ";
	const string C_PROGRAM_INFO_NA          = "N/A";

private:
	
	// ...

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Data type aliases
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

	// ...

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Functions.
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

	// Constructors and destructors

	TestApplication ( int argc, char* argv [] );
	~TestApplication ();

	// Overriden functions.

	virtual void command_interpreter ( vector <string> command_line ) override;

	// Application commands.

	void command_main_program ();
	void command_test_1       ();
	void command_test_2       ();
	void command_test_3       ();
	void command_test_4       ();
	void command_test_5       ();
	
	// Display functions.

	void print_program_info ();	
	void print_program_info ( string class_name, string function_name, string description );	
	void print_vector       ( vector <double> v, int precision, string label, string vector_name );
	
	// Loggin functions.

	string log_to_string ( long long time_point, long long time_stop, string message );
		
private:

	// ...

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Accessors and mutators
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------

	// Accessors

	// ...

	// Mutators.

	// ...

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Member data.
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

	// ...
	
private:

	// ...
};


//--------------------------------------------------------------------------------------------------
// Project:	    Generic Console Application Platform (G-CAP).
// Application: N/A
// Class:	    ConsoleApplication
// Version:	    1.3
// Date:	    2014
// Author:	    Rohin Gosling
//
// Description:
//
//   Generic console application platform.
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
//--------------------------------------------------------------------------------------------------

#pragma once

// Export managment

#ifdef CONSOLE_APPLICATION_EXPORTS
	#define CONSOLE_APPLICATION_API __declspec ( dllexport ) 
#else
	#define CONSOLE_APPLICATION_API __declspec ( dllimport ) 
#endif

// Include files.

#include "stdafx.h"

#include <string>
#include <vector>

// Explicate STL template class exports.

template class CONSOLE_APPLICATION_API std::basic_string<char,std::char_traits<char>,std::allocator<char>>;
template class CONSOLE_APPLICATION_API std::vector<std::string>;

// CLass decleration.

using namespace std;

namespace common
{
	class CONSOLE_APPLICATION_API ConsoleApplication
	{
		//--------------------------------------------------------------------------------------------------
		// Data type aliases
		//--------------------------------------------------------------------------------------------------

	public:

		// ...

		//--------------------------------------------------------------------------------------------------
		// Constants
		//--------------------------------------------------------------------------------------------------

	public:

		// ...

	protected:

		const int C_DEFAULT_MARGIN = 0;

	private:
	
		const string C_MSG_EXECUTION_TIME    = "Program execution time:";		
		const string C_MSG_EXCEPTION         = "EXCEPTION: ";
		const string C_MSG_EXCEPTION_GENERIC = "Generic exception.";
	
	
		//--------------------------------------------------------------------------------------------------
		// Functions.
		//--------------------------------------------------------------------------------------------------

	public:

		// Constructors and destructors

		ConsoleApplication  ();								// Default constructor.
		ConsoleApplication  ( int argc, char* argv [] );	// Pass through the command line arguments.
		~ConsoleApplication ();								// Desctructor.

		// General consumer functions.

		void initialize    ( int argc, char* argv [] );		// Used to initialize constructors. May also be used indepementantly.
		void run           ();								// Launches the applcation.
		void clear_console ();								// Clear the console.
		void move_to       ( int x, int y );				// Move the cursor to the location specified by (x,y);
	
	private:

		// Cammand line management functions.

		vector <string> get_command_line_arguments ( int argc, char* argv [ ] );
		virtual void    command_interpreter        ( vector <string> command_line ) = 0;

		// display functions.

		string print_execution_time ( long long time );
		string print_execution_time ( int margin, long long time );

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

		int margin { C_DEFAULT_MARGIN };		// Initialize default text margin.

	private:

		vector <string> command_line;			// Pupulated using GetCommandLineArguments.

	
	};
}



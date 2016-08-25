//--------------------------------------------------------------------------------------------------------------------------------------
// Project:		Generic Console Application Platform (G-CAP).
// Application:	N/A
// Class:       StaticUtility
// Version:		1.3
// Date:        2015-07-10
// Author:		Rohin Gosling
//
// Description:
//
//   General purpose functions for console applications.
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

#ifdef STATIC_UTILITY_EXPORTS
	#define STATIC_UTILITY_API __declspec ( dllexport ) 
#else
	#define STATIC_UTILITY_API __declspec ( dllimport ) 
#endif

#include "stdafx.h"

#include <string>
#include <vector>
#include <iomanip>
#include <chrono>

//template class STATIC_UTILITY_API std::basic_string<char,struct std::char_traits<char>,class std::allocator<char>>;

using namespace std;
using namespace std::chrono;

namespace common
{
	class STATIC_UTILITY_API StaticUtility
	{
		//----------------------------------------------------------------------------------------------------------------------------------
		// Data type aliases
		//----------------------------------------------------------------------------------------------------------------------------------

	public:

		using Natural      = unsigned long;										// N: n Є {  0 ≤ x ≤ ∞  }
		using Integer      = long;												// Z: n Є { -∞ ≤ x ≤ ∞  }		
		using Real         = double;											// R:
		using Complex	   = struct { Real r; Real i; };						// C:
		using Rational     = struct { Integer n; Integer d; };					// Q:
		using Alignment    = enum { ALIGN_LEFT, ALIGN_CENTER, ALIGN_RIGHT };	// Text alignment.
		using TokenList    = vector <string>;									// List of token strings.
		using Time         = unsigned long long;								// Scalar time, in miliseconds.


		//----------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		//----------------------------------------------------------------------------------------------------------------------------------

	public:

		// Constructors and destructors

		StaticUtility ();		// Default constructor.	
		~StaticUtility ();		// Desctructor.

		// General purpose, usefull functions.

		static void		 sleep                  ( const int milliseconds );
		static Natural	 get_max_length         ( vector < string > string_list );
		static string	 to_upper               ( string input_string );
		static string	 to_lower               ( string input_string );
		static string    replace_string         ( string input_string, const string& search, const string& replace );
		static string    text_field             ( string field_label, string field_value, int label_width, string delimiter );
		static TokenList tokenize_csv_string    ( string input_string );
		static string    int_pad_left           ( int i, char ch, int width );
		static string    get_numerator_string   ( string ratio_string );
		static string    get_denominator_string ( string ratio_string );

		// Date and time.

		static Time   get_time             ();											// TODO:
		static Time   get_time_elapsed     ( Time time_start );							// TODO:
		static Time   time_point_to_int    ( steady_clock::time_point chrono_time );	// TODO:

		static string time_to_string       ( Time time );
		static string time_to_string       ( Time time, bool show_hours, bool show_minutes, bool show_seconds, bool show_milliseconds );
		static Time   time_to_milliseconds ( Time time );
		static Time   time_to_seconds      ( Time time );
		static Time   time_to_minutes      ( Time time );
		static Time   time_to_hours        ( Time time );

		// Random number generation.

		static void   random_seed ();
		static double random_between ( double a, double b );

		// String conversion functions.

		static string align_string      ( string text, Integer width, Alignment alignment );
		static string to_aligned_string ( Integer value, Integer width, Alignment alignment );
		static string to_aligned_string ( Real value, Integer precision, Integer width, Alignment alignment );


		//--------------------------------------------------------------------------------------------------------------------------------------
		// Accessors and mutators
		//--------------------------------------------------------------------------------------------------------------------------------------

		// Accessors

		// ...

		// Mutators.

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------
		// Constants
		//--------------------------------------------------------------------------------------------------------------------------------------

	public:

		// ...

	private:

		// ...

	};
}


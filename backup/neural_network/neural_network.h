//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Project:	    N/A
// Application: Neural Network Class
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

#ifdef NEURAL_NETWORK_EXPORTS
	#define NEURAL_NETWORK_API __declspec ( dllexport ) 
#else
	#define NEURAL_NETWORK_API __declspec ( dllimport ) 
#endif

// Include files.

#include "stdafx.h"
#include "neural_network_layer.h"

#include <iostream>
#include <string>
#include <vector>

// Class

using namespace std;

template class NEURAL_NETWORK_LAYER_API std::vector<common::NeuralNetworkLayer,std::allocator<common::NeuralNetworkLayer>>; 

namespace common
{
	class NEURAL_NETWORK_API NeuralNetwork
	{
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Constants
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		const double C_DEFAULT_WEIGHT = 0.0;
		const double C_DEFAULT_BIAS   = 0.0;

	private:

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Data type aliases
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		// ...

	private:

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Functions.
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		// Constructors and destructors

		NeuralNetwork ();
		NeuralNetwork ( vector<int> network );
		NeuralNetwork ( vector<int> network, double wieght, double bias );
		NeuralNetwork ( const NeuralNetwork& element );		// Copy constructor.
		~NeuralNetwork ();

		// Operators.

		NeuralNetwork& operator = ( const NeuralNetwork& element );

		// Network functions.

		void           initialize             ( vector<int> network );
		void           initialize             ( vector<int> network, double weight, double bias );
		void           set_weights            ( double weight );
		void           randomize_weights      ( double min, double max );		
		void           set_bias_vectors		  ( double bias_term );
		void           randomize_bias_vectors ( double min, double max );
		vector<double> compute_network        ( vector<double> x );
		double         back_propagation       ( vector<vector<double>> x, vector<vector<double>> y, double maxE, double epoch_max, long long time_out );

		// Display functions.

		string to_string       ();
		string to_string_int8  ();	// TODO:
		string to_string_uint8 ();	// TODO:
		string to_string       ( int precision, bool header_row_enabled, bool activation_function_enabled );
		
	private:

		// ...

		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		// Member data.
		//
		// Note:
		// - We are not using accessors and mutators in vafor of moden C++ best practices.
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	public:

		vector<NeuralNetworkLayer> neural_network;		// Array of neural network layers, out of which the neural network is comprised. This is the main neural network data entity.
	
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

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------------------------------------------------
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
//   - Ported from my old Java version of this class.
//     The Java version was it's self, ported from an even older functional C version.
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
//--------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "stdafx.h"
#include "neural_network.h"

#define ACTIVATION_FUNCTION_TYPE NeuralNetworkLayer::ActivationFunctionType

namespace common
{
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Constructor/s
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// Constructor 1/3.
	
	NeuralNetwork::NeuralNetwork ()
	{
		initialize ( (vector<int>) (0) );
	}

	// Constructor 2/3.

	NeuralNetwork::NeuralNetwork ( vector<int> network_architecture )
	{
		initialize ( network_architecture );
	}

	// Constructor 3/3.

	NeuralNetwork::NeuralNetwork ( vector<int> network_architecture, double weight, double bias )
	{
		initialize ( network_architecture, weight, bias );
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Copy constructor/s.
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	NeuralNetwork::NeuralNetwork ( const NeuralNetwork& element )
	{
		this->neural_network = element.neural_network;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Destructor
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	NeuralNetwork::~NeuralNetwork ()
	{
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Operator/s.
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// Operator: Assignment.
	// Symbol:   =

	NeuralNetwork& NeuralNetwork::operator = ( const NeuralNetwork& element )
	{
		this->neural_network = element.neural_network;

		return *this;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - initialize
	//
	// Description:
	//
	// - Initialize the neural network parameters and perceptron architecture. 
	//
	// Arguments:   
	//
	// - network
	//   And array of integers, representing an architecture vector, where each element of the vector, specifies the number of perceptron units in
	//   consecutive layers of the network.
	//
	//   Example 1:
	//
        //     Given: Neural network architecture vector, network = [ 2 2 1 ]
	//
        //     Then:  One input layer, with 2 input units, and 2 output units.    
        //            One output layer, with 2 input units, and 1 output unit.
	//
	//   Example 2:
	//
        //     Given: Neural network architecture vector, network = [ 4 8 6 2 ]
	//
        //     Then:  One input layer, with 4 input units, and 8 output units.
        //            One hidden layer, with 8 input units, and 6 output units.
        //            One output layer, with 6 input units, and 2 output units.
	//
	//   Note:
	//
	//   - Teaditionaly, the word layer is used to refer to a layer of perceptron units.
	//
	//   - In our case here, it is mathematicaly convenient to express a layer, as a paired combination of input perceptron units, and 
	//     output perceptron units, in order to focus computing atention on the weights between the input and output units.
	//
	//   - Within this framework of terminology, a traditional 3-layer perceptron network, with one input layer, one hidden layer and one
	//     output layer, i.e. network = [ i h o ], is now expressed as a 'two' layer network, with one input layer, and one output layer.
	//     Where the input layer has i input units, and h output units. And the output layer has h input units, and o output units.
	//
	//   - While this is done for the sake of internal mathematical convenience, a comsumer of NeuralNetowrk, will interact with the class
	//     using the traditional terminology and architecture.
	//     i.e. 
	//     A comsumer of NeuralNetwork, calls NeuralNetwork::Initialize ( (vector<double>) { 2, 4, 1 } ), to describe a traditional
	//     3-layer architecture, with one input layer of 2 units, one hidden layer of 4 units, and one output layer of 1 unit.
	//     Internaly however, the NeuralNetworkLayer class, will convert this abstract architecture, into a phisical architecture, with
	//     only 'two' layers; an input layer with 2 inputs and 4 outputs, and an output layer with 4 input units and 1 output unit.
	//
	// Return Value:
	//
	// - N/A
	//
	// Pre-Conditions:
	//
	// - N/A
	//   
	// Post-Conditions:
	//
	// - The neural netowrk's parameters and architecture have been initialized to the specified values.
	//
	//   Note:
	//   - Calling initialize, efectivly resets the netowrk, including resetting all learned weight and bias vectors, along with resetting
	//     cached input/output vectors.
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	// Overload 1/2

	void NeuralNetwork::initialize ( vector<int> network_architecture )
	{
		initialize ( network_architecture, C_DEFAULT_WEIGHT, C_DEFAULT_BIAS );
	}

	// Overload 2/2

	void NeuralNetwork::initialize ( vector<int> network_architecture, double weight, double bias )
	{
		// Initialize local constants.

		// Initial weight and bias ranges

		const int            min = 0;
		const int            max = 1;
		const vector<double> initial_bias_range   { -0.5, 0.5 };
		const vector<double> initial_weight_range { -0.5, 0.5 };
		
		// Initial activastion function.

		const ACTIVATION_FUNCTION_TYPE activation_function_type_hidden = ACTIVATION_FUNCTION_TYPE::TANH;
		const ACTIVATION_FUNCTION_TYPE activation_function_type_output = ACTIVATION_FUNCTION_TYPE::LINEAR;

		// Initialize local variables.

		int input_count  { 0 };
		int output_count { 0 };

		// Calculate the number of layers.
		//
		// - This is the number of weight layers, as aposed to the number of perceptron layers.
		// - Example: 
		//   For three layers of perceptrons, network = [ i h o ], there are two layers of weight vectors, W = [ ih ho ]

		int layer_count = network_architecture.size () - 1;

		this->neural_network.resize ( layer_count );

		// Initialize all layers in the network.

		for ( int i = 0; i < layer_count; i++ )
		{
			// Initialize the number of input and output perceptrons in this layer.

			input_count  = network_architecture [ i     ];
			output_count = network_architecture [ i + 1 ];

			this->neural_network [ i ].initialize ( input_count, output_count );

			// Configure initial layer weight vectors and initial bias vector.

			this->neural_network [ i ].set_weights     ( weight );
			this->neural_network [ i ].set_bias_vector ( bias   );

			// Configure activation function.
			//
			// - Use the hidden layer activation function for layers 0 through layer_count - 2.  i.e. ( i <  layer_count - 1 )
			// - Use the output layer activation function for layer layercount - 1.              i.e. ( i >= layer_count - 1 )

			if ( i < layer_count - 1 )
			{
				this->neural_network [ i ].activation_function_type = activation_function_type_hidden;
			}
			else
			{
				this->neural_network [ i ].activation_function_type = activation_function_type_output;
			}			
		}
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// set_weights
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void NeuralNetwork::set_weights ( double weight )
	{
		// Retrieve the current number of layers.

		int layer_count { (int) this->neural_network.size() };

		// Set all weights, of all perceptron units, of all layers, to the value specified by argument weight.

		for ( int layer_index = 0; layer_index < layer_count; layer_index++ )
		{
			this->neural_network [ layer_index ].set_weights ( weight );
		}
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// randomize_weights
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void NeuralNetwork::randomize_weights ( double min, double max )
	{
		// Retrieve the current number of layers.

		int layer_count { (int) this->neural_network.size() };

		// Set all weights, of all perceptron units, of all layers, to the value specified by argument weight.

		for ( int layer_index = 0; layer_index < layer_count; layer_index++ )
		{
			this->neural_network [ layer_index ].randomize_weights ( min, max );
		}
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// setbias_vectors
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void NeuralNetwork::set_bias_vectors ( double bias_term )
	{
		// Retrieve the current number of layers.

		int layer_count { (int) this->neural_network.size() };

		// Set all bias vectors, of all layers, to the value specified by argument weight.

		for ( int layer_index = 0; layer_index < layer_count; layer_index++ )
		{
			this->neural_network [ layer_index ].set_bias_vector ( bias_term );
		}
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// randomize_bias_vectors
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	void NeuralNetwork::randomize_bias_vectors ( double min, double max )
	{
		// Retrieve the current number of layers.

		int layer_count { (int) this->neural_network.size() };

		// Set all bias vectors, of all layers, to the value specified by argument weight.

		for ( int layer_index = 0; layer_index < layer_count; layer_index++ )
		{
			this->neural_network [ layer_index ].randomize_bias_vector ( min, max );
		}
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - compute_network
	//
	// Description:
	//
	// - Takes an input vector X, and computes the output vector Y, based on the current weight vectors for each layer in the network.
	//
	// Arguments:   
	//
	// - x
	//   - Real value input vector.	
	//
	// Return Value:
	//
	// - compute_network outputs the neural network output vector y, as influenced by the current weaight vectors for each layer in the network.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   - The neural network needs to be initialized. However, the default state configured by the constructors, leaves the network in an initialized
	//     condition. Therefore, compute_network may effectivly be called at any time after an instance of NeuralNetwork has been created. 
	//   - The neuarl network does not need to be trained, before calling compute_network either. Calling compute_network on an untrained, or partialy
	//     trained netowrk, may simply result in undefined and/or incoherant behavior.
	//   
	// Post-Conditions:
	//
	// - Output Vector:
	//   The neural networks output vector is computed and returned to the caller of compute_network.
	//
	// - Internal layer Outputs:
	//   Aside from the output vector, all internal layer output vectors, are buffered in their respective layer outputs buffers, for use with the 
	//   backpropagation algorythm, and/or any other algorythms that may require historical access to a computed netowrks internal layer output vectors.
	//
	// - Note:
	//   For the sake of reducing memory requirements for scenarios where internal layer output vectors are not required, such as deployed trained
	//   networks what require no further learning is required, then buffering of internal layer output vectors may be turned off by setting 
	//   "buffer_internal_outputs" to false. "buffer_internal_outputs" is set to true, by default.
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	vector<double> NeuralNetwork::compute_network ( vector<double> x )
	{
		// Initialize local variables.

		vector<double> y;
		int            layer_count = this->neural_network.size ();

		// Feed forward through the network, and compute the inputs and outputs of each layer.

		for ( int i = 0; i < layer_count; i++ )
		{
			y = this->neural_network [ i ].compute_layer ( x );		// The out put of layer i, given input x, is y.
			x = y;													// The input of layer i+1, will be the output of layer i, therfore, x = y.
		}

		// Return final output vector to caller.		

		return y;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - back_propagation
	//
	// Description:
	//
	// - Stochastic backpropagation algorythm.
	//
	// Arguments:   
	//
	// - x
	//   Training input vector. Sample input.
	//
	// - y
	//   Training output vector. Desired target output, for given sample input vector.
	//
	// - maxE
	//   The maximum alowable mean squared error (MSE), at which to terminate further training.
	//   Once an epcoh is reached where the maximum mean squared error (maxE) is observed during that epoch, then terminate training.
	//
	// - epoch_max
	//   The maximum number of epochs over which to train.
	//   Once we have reached epoch epoch_max-1, then terminate training.
	//
	// - time_out
	//   The maximum period of time over which to train, measured in miliseconds.
	//   Once elapsed time is greater than or equal to time_out miliseconds, then terminiate traing.
	//
	// - Note:
	//
	//   - Training Termination Events:
	//
	//     Training may terminate on any one of the three events, listed in order of presedence;
	//
	//     1. time_out: 
	//
	//        Training is terminiated, when the elapsed training time is greater than or equal to time_out, measured in miliseconds.
	//        We terminate training on this event, regardless of the maximum number of epochs, or the maximum mean squared error.
	//        - Training time out may be disabled by setting time_out to zero miliseconds.
	//          In this case, the training time out will be ignored and traiining will continue forever, short of any other terminating events.
	//
	//     2. epoch_max:
	//
	//        Training is terminiated, when the maximum number of epochs has been reached.
	//        We terminate training on this event, if the elapsed time is still less than time_out, and regardless of the maximum mean squared error.
	//        - The maximum epoch may be disabled by setting epoch_max to zero. 
	//          In this case, the number of training epochs will be ignored, and training will contrinue for an infinite number of epochs, or until 
	//          any other terminating event is raised.
	//
	//     3. maxE:
	//
	//        Training is terminated, when the maximum mean square error was observed during an epoch.
	//        We terminate training on this event, if the elapsed time is still less than time_out, and the maximum epoch has not yet been reached, 
	//        and an epoch has been reached, where the mean squared error remained at or below maxE.
	//        - The maximum mean squared error can not be disabled, however, setting maxE to zero, will cause training to continue until there is
	//          zero error, which in most cases will be imposiable, and therefore the training algorythm will spin for ever, or until any other 
	//          terminaing event is raised.
	//
	//
	// Return Value:
	//
	// - The algorythm returnes the maxumum mean squared error (MSE).	
	//
	// Pre-Conditions:
	//
	// - The x and y (input and target) training vectors, should be non-zero in leangth. i.e. there should be some training data.
	//   Note:
	//   - The algorythm will opearte on what ever training data is supplied through the x and y (input and target) training vectors.
	//   - It is up to the supplier of the training to data, to supply training data that actualy represents convergable data samples.
	//   - The algorythm will attempt to generalize across what ever data it is given, even if the data is too systemicly incoherant and
	//     chaotic for practical convergence to be achieved.
	//   
	// Post-Conditions:
	//
	// - The internal wieght vectors and bias vecotrs for each layer in the netowrk, have been updated to reflect the best posible generalization
	//   across the supplied training data, given the specific traing examples supplied, and the amount of time, epochs, and target error specified.
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	double NeuralNetwork::back_propagation ( vector<vector<double>> x, vector<vector<double>> y, double maxE, double epoch_max, long long time_out )
	{
		// while terminating event not raised, do

			// For all {x,y) in training_data, do

				// 1. Feed forward.
				//    y' = compute_network ( x );
				//    y  = training_data   ( x );

				// 2. Compute errors.
				//    e = y' - y

				// 3. Compute weight deltas
				//    - Compute ∆wₒ for outut layer.
				//    - Compute ∆wᵢ for all other layers
		
				// 4. Update network weights.

		return 0;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - to_string
	//
	// Description:
	//
	// - Console representation of the neural network.
	//
	// Arguments:   
	//
	// - N/A
	//
	// Return Value:
	//
	// - Human readable string representaton of the neural network.
	//
	//   Note:
	//   For large networks, this may be an enormous string.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   to_string may be called regardless of the state of the neural network. The output string sill simply reflect what ever state the network is in.
	//   
	// Post-Conditions:
	//
	// - A human readable string repreentation of the neural network has been generated. and returned.
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	string NeuralNetwork::to_string ()
	{
		return nullptr;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - function_name
	//
	// Description:
	//
	// - Function description.
	//
	// Arguments:   
	//
	// - x
	//   Description of argument x.
	//
	// Return Value:
	//
	// - Description of return value.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   
	// Post-Conditions:
	//
	// - N/A
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	string NeuralNetwork::to_string_int8 ()
	{
		return nullptr;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - function_name
	//
	// Description:
	//
	// - Function description.
	//
	// Arguments:   
	//
	// - x
	//   Description of argument x.
	//
	// Return Value:
	//
	// - Description of return value.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   
	// Post-Conditions:
	//
	// - N/A
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	string NeuralNetwork::to_string_uint8 ()
	{
		return nullptr;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - function_name
	//
	// Description:
	//
	// - Function description.
	//
	// Arguments:   
	//
	// - x
	//   Description of argument x.
	//
	// Return Value:
	//
	// - Description of return value.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   
	// Post-Conditions:
	//
	// - N/A
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

	string NeuralNetwork::to_string ( int precision, bool header_row_enabled, bool activation_function_enabled )
	{
		// Initialize constants.

		const string NEW_LINE = "\n";

		// Initialize local variables.

		string s { "" };

		// Build string.

		int layer_count = this->neural_network.size ();

		for ( int i = 0; i < layer_count; i++ )
		{	
			s += this->neural_network [ i ].to_string ( precision, header_row_enabled, activation_function_enabled );

			if ( i < layer_count - 1 )
			{
				s += NEW_LINE;
			}
		}

		// Return string to caller.

		return s;
	}

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	// Function:
	//
	// - function_name
	//
	// Description:
	//
	// - Function description.
	//
	// Arguments:   
	//
	// - x
	//   Description of argument x.
	//
	// Return Value:
	//
	// - Description of return value.
	//
	// Pre-Conditions:
	//
	// - N/A
	//   
	// Post-Conditions:
	//
	// - N/A
	//
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
}

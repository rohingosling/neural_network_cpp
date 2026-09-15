//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 5.1
// Date:    2014-08-02
// Author:  Rohin Gosling
//
// Description:
//
//   Implementation of the NeuralNetwork class. A general-purpose, fully connected feed-forward neural network
//   supporting both regression and classification.
//
//   This file defines weight initialisation, the forward and backward propagation passes, mini-batch 
//   gradient-descent training, loss computation, inference (Predict and Classify), and the enum-to-string and 
//   formatting helpers.
//
// TODO:
//
//   1. Add support for additional weight-initialization methods (Xavier/Glorot for tanh and sigmoid, He for ReLU).
//   2. Implement the Cross-Entropy loss function, the Adam optimizer, and model save/load (currently scaffolded).
//   3. Add a log file.
//
//---------------------------------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------------------------------
// Background, Usage, and Version History
//
// Usage:
//
//   Regression Problems :
//
//   - Hidden Layers:
//
//     - The most common activation function for hidden layers in regression problems is the Rectified Linear
//       Unit ( ReLU ).
//     - ReLU allows gradients to flow back efficiently during training ( avoiding vanishing gradients ) and can
//       model non-linear relationships well.
//     - Other options include Leaky ReLU, which addresses the "dying ReLU" problem, or even parametric ReLU ( PReLU )
//       for added flexibility.
//
//   - Output Layer:
//
//     - In regression, the output layer typically uses a linear activation function.
//     - This ensures the final output represents a continuous value on the real number line, which aligns with the
//       desired outcome in regression tasks ( e.g., predicting house prices or stock values ).
//
//   Classification Problems :
//
//   - Hidden Layers:
//
//     - Similar to regression, ReLU is a popular choice for hidden layers in classification problems due to its
//       efficiency in training. However, other options like tanh ( squashes values between - 1 and 1 ) or sigmoid
//       ( outputs between 0 and 1 ) can also be used. These functions can be helpful if your data naturally falls
//       within a specific range.
//
//     - Output Layer:
//
//       - The choice for the output layer depends on the number of classes you're predicting.
//
//       - Binary Classification ( Two Classes ):
//         - Use the sigmoid function to transform the final output into a probability between 0 ( class 1 )
//           and 1 ( class 2 ).
//
//       - Multi-Class Classification ( More Than Two Classes ):
//         - Here, you typically employ the softmax function.
//         - Softmax normalizes the output layer's activations into probabilities that sum to 1, representing the
//           probability of each class.
//
//   - Note:
//
//     - The "optimal" activation function can vary depending on your specific dataset and problem.
//
//     - Experimentation is often key. It's a good practice to start with the recommended choices above and then compare
//       different options using techniques like grid search or random search to find the best-performing combination
//       for your situation.
//
//---------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

#include "neural_network.h"
#include "math_ai.h"

using namespace std;

//=====================================================================================================================
// Constructors
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
// Constructor 1/1: NeuralNetwork
//
// Description:
//
//   Constructs a neural network with the given architecture and hyperparameters, and immediately initialises its
//   weights and biases to small random values. The network is ready to be trained once constructed.
//
// Arguments:
//
//   layers (const vector <int>&):
//     The number of neurons in each layer, from input to output. For example, { 2, 3, 1 } describes a network with two
//     inputs, one hidden layer of three neurons, and a single output.
//
//   activation_functions (const vector <ActivationFunction>&):
//     The activation function applied at each layer after the input layer, with one entry per non-input layer.
//
//   loss_function (LossFunction):
//     The loss function used to measure training error.
//
//   learning_rate (double):
//     The step size applied to the gradients when updating weights and biases.
//
//   epoch_count (int):
//     The number of complete passes over the training data.
//
//   batch_size (int):
//     The number of training samples processed per mini-batch before the weights are updated.
//
//   optimization_algorithm (OptimizationAlgorithm):
//     The weight-update strategy used during training.
//
//   training_results_file_name (const string):
//     Path of the CSV file to which per-epoch loss is written during training.
//
//---------------------------------------------------------------------------------------------------------------------

NeuralNetwork::NeuralNetwork
(
    const vector <int>&                layers,
    const vector <ActivationFunction>& activation_functions,
    LossFunction                       loss_function,
    double                             learning_rate,
    int                                epoch_count,
    int                                batch_size,
    OptimizationAlgorithm              optimization_algorithm,
    const string                       training_results_file_name
) : m_layers                     ( layers ),
    m_activation_functions       ( activation_functions ),
    m_loss_function              ( loss_function ),
    m_learning_rate              ( learning_rate ),
    m_epoch_count                ( epoch_count ),
    m_batch_size                 ( batch_size ),
    m_optimization_algorithm     ( optimization_algorithm ),
    m_training_results_file_name ( training_results_file_name )
{
    InitializeWeights ();
}

//=====================================================================================================================
// Methods
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
// Method: InitializeWeights
//
// Description:
//
//   Initialises the weight and bias vectors for every layer with random values drawn from a uniform distribution
//   over [-1, 1]. Random initialisation breaks symmetry so the neurons can learn distinct features.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::InitializeWeights ()
{
    default_random_engine              generator;
    uniform_real_distribution <double> distribution ( -1.0, 1.0 );

    for ( size_t i = 0; i < m_layers.size () - 1; ++i )
    {
        int layer_size      = m_layers [ i ];
        int layer_size_next = m_layers [ i + 1 ];

        MathAI::Vector layer_weight_vector ( layer_size_next * layer_size );
        MathAI::Vector layer_bias_vector   ( layer_size_next );

        for ( double& weight : layer_weight_vector ) weight = distribution ( generator );
        for ( double& bias   : layer_bias_vector   ) bias   = distribution ( generator );

        m_weight_vectors.push_back ( layer_weight_vector );
        m_bias_vectors.push_back   ( layer_bias_vector   );
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Train
//
// Description:
//
//   Trains the network on the supplied data using backpropagation and mini-batch gradient descent. For each epoch
//   the data is processed batch by batch: gradients are accumulated from forward and backward passes, and the
//   weights and biases are then updated. The epoch loss is logged to the results file and printed to the console.
//
// Arguments:
//
//   training_data_x (const MathAI::Matrix&):
//     The input features. Each row is one training sample and each column is one feature.
//
//   training_data_y (const MathAI::Matrix&):
//     The target outputs. Each row holds the expected output values for the corresponding input sample.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::Train ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y )
{
    ShowCursor ( false );

    // Initialise local variables.

    size_t epoch_count         = m_epoch_count;                             // Number of training epochs.
    size_t batch_sample_count  = m_batch_size;                              // Number of samples per batch.
    size_t global_sample_count = training_data_x.size ();                   // Total number of samples in the global training dataset.
    size_t batch_count         = global_sample_count / batch_sample_count;  // Compute the number of batches per epoch.

    // Open the CSV file for writing training results and add CSV header row to the file.

    ofstream results_file ( m_training_results_file_name );

    if ( !results_file.is_open () )
    {
        cerr << "Error: Unable to open training results file: " << m_training_results_file_name << endl;
        return;
    }

    results_file << "epoch_index,loss\n";

    // Train the neural network over the specified number of epochs.

    for ( size_t epoch_index = 0; epoch_index < epoch_count; ++epoch_index )
    {
        // Itterate over epoch training batches.

        for ( size_t batch_index = 0; batch_index < batch_count; ++batch_index )
        {
            MathAI::Matrix weight_gradients ( m_weight_vectors.size () );     // Initialize weight gradients for this batch.
            MathAI::Matrix bias_gradients   ( m_bias_vectors.size   () );     // Initialize bias gradients for this batch.

            InitializeGradients ( weight_gradients, bias_gradients );         // Set gradients to zero.

            // Iterate over each sample in the current training batch.

            for ( int sample_index = 0; sample_index < batch_sample_count; ++sample_index )
            {
                size_t   batch_sample_index    = batch_index * batch_sample_count + sample_index;                              // Calculate the global sample index.
                MathAI::Matrix activations     = PropagateForward  ( training_data_x [ batch_sample_index ] );                 // Perform forward pass.
                MathAI::Matrix error_gradients = PropagateBackward ( activations, training_data_y [ batch_sample_index ] );    // Perform backward pass.

                AccumulateGradients ( activations, error_gradients, weight_gradients, bias_gradients );                        // Accumulate gradients.
            }

            UpdateWeightsAndBiases ( weight_gradients, bias_gradients );  // Update weights and biases using accumulated gradients.
        }

        // Compute loss.
        // - Write epoch loss to results file.
        // - Write epoch loss to terminal file.

        double loss = ComputeLoss ( training_data_x, training_data_y );
        results_file << epoch_index << "," << loss << "\n";
        cout << TrainingProgressToString ( epoch_index, epoch_count, loss ) << flush;
    }

    // Clean up and exit training.
    // - Close training results file.
    // - Enable the terminal cursor and write a new line to the terminal.

    results_file.close ();
    ShowCursor ( true );
    cout << endl;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: InitializeGradients
//
// Description:
//
//   Resizes the per-layer weight and bias gradient vectors to match the network's parameters and sets every element
//   to zero. Called at the start of each mini-batch to reset the gradient accumulators.
//
// Arguments:
//
//   weight_gradients (MathAI::Matrix&):
//     The weight-gradient accumulator to reset, sized to match the weight vectors.
//
//   bias_gradients (MathAI::Matrix&):
//     The bias-gradient accumulator to reset, sized to match the bias vectors.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::InitializeGradients ( MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients )
{
    // Initialize gradients for the current training batch.

    for ( size_t layer_index = 0; layer_index < m_weight_vectors.size (); ++layer_index )
    {
        weight_gradients [ layer_index ].resize ( m_weight_vectors [ layer_index ].size (), 0.0 );
        bias_gradients   [ layer_index ].resize ( m_bias_vectors   [ layer_index ].size (), 0.0 );
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: AccumulateGradients
//
// Description:
//
//   Adds the contribution of a single sample to the running weight and bias gradient accumulators, combining the
//   sample's activations and error gradients. Called once per sample within a mini-batch during training.
//
// Arguments:
//
//   activations (const MathAI::Matrix&):
//     The per-layer activations from the sample's forward pass.
//
//   error_gradients (const MathAI::Matrix&):
//     The per-layer error gradients from the sample's backward pass.
//
//   weight_gradients (MathAI::Matrix&):
//     The weight-gradient accumulator to add into.
//
//   bias_gradients (MathAI::Matrix&):
//     The bias-gradient accumulator to add into.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::AccumulateGradients
(
    const MathAI::Matrix& activations,
    const MathAI::Matrix& error_gradients,
          MathAI::Matrix& weight_gradients,
          MathAI::Matrix& bias_gradients
)
{
    for ( size_t layer_index = 1; layer_index < m_weight_vectors.size (); ++layer_index )
    {
        for ( size_t neuron_index = 0; neuron_index < m_weight_vectors [ layer_index ].size (); ++neuron_index )
        {
            size_t layer_neuron_count_next = m_layers [ layer_index + 1 ];
            size_t activation_index        = neuron_index % layer_neuron_count_next;

            if ( activation_index < activations [ layer_index ].size () )
            {
                double neuron_gradient                           = error_gradients [ layer_index ][ neuron_index ];
                double previous_layer_neuron_activation          = activations [ layer_index ][ activation_index ];
                weight_gradients [ layer_index ][ neuron_index ] = neuron_gradient * previous_layer_neuron_activation;
            }
            else
            {
                cerr << "[Train] ERROR:" << endl
                     << "Index out of bounds : activation_index = " << activation_index                    << endl
                     << "activations[layer_index].size() = "        << activations [ layer_index ].size () << endl;
            }
        }

        for ( size_t neuron = 0; neuron < m_bias_vectors [ layer_index ].size (); ++neuron )
        {
            double neuron_error_gradient              = error_gradients [ layer_index + 1 ][ neuron ];
            bias_gradients [ layer_index ][ neuron ] += neuron_error_gradient;
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: UpdateWeightsAndBiases
//
// Description:
//
//   Applies the accumulated mini-batch gradients to the network's weights and biases, stepping each parameter
//   against its gradient scaled by the learning rate and divided by the batch size.
//
// Arguments:
//
//   weight_gradients (const MathAI::Matrix&):
//     The accumulated weight gradients for the batch.
//
//   bias_gradients (const MathAI::Matrix&):
//     The accumulated bias gradients for the batch.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::UpdateWeightsAndBiases ( const MathAI::Matrix& weight_gradients, const MathAI::Matrix& bias_gradients )
{
    // Initialise local variables.

    size_t weight_vector_count = m_weight_vectors.size ();

    // update weights and biases.

    for ( size_t layer_index = 0; layer_index < weight_vector_count; ++layer_index )
    {
        size_t layer_weight_vector_count = m_weight_vectors [ layer_index ].size ();
        size_t layer_bias_vector_count   = m_bias_vectors   [ layer_index ].size ();

        for ( size_t neuron_index = 0; neuron_index < layer_weight_vector_count; ++neuron_index )
        {
            double neuron_weight_gradient                     = weight_gradients [ layer_index ][ neuron_index ];
            m_weight_vectors [ layer_index ][ neuron_index ] -= m_learning_rate * neuron_weight_gradient / m_batch_size;
        }

        for ( size_t neuron_index = 0; neuron_index < layer_bias_vector_count; ++neuron_index )
        {
            double neuron_bias_gradient                     = bias_gradients [ layer_index ][ neuron_index ];
            m_bias_vectors [ layer_index ][ neuron_index ] -= m_learning_rate * neuron_bias_gradient / m_batch_size;
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ShowCursor
//
// Description:
//
//   Shows or hides the console text cursor by emitting the appropriate ANSI escape sequence. Used to hide the
//   cursor while the training-progress line is being repeatedly overwritten.
//
// Arguments:
//
//   cursor_visible (bool):
//     When true, the cursor is shown; when false, it is hidden.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::ShowCursor ( bool cursor_visible )
{
    if ( cursor_visible )
    {
        cout << "\033[?25h";
    }
    else
    {
        cout << "\033[?25l";
    }

    cout << flush;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: PropagateForward
//
// Description:
//
//   Performs a forward pass through the network for a single input sample, computing and returning the activations
//   of every layer. For each neuron the weighted sum of the previous layer's activations plus a bias is passed
//   through the layer's activation function.
//
// Arguments:
//
//   input (const MathAI::Vector&):
//     The input feature vector for one sample.
//
// Returns:
//
//   MathAI::Matrix:
//     The activations of every layer, with the input as the first row and the output-layer activations last.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Matrix NeuralNetwork::PropagateForward ( const MathAI::Vector& input )
{
    MathAI::Matrix activations;
    activations.push_back ( input );

    for ( size_t i = 0; i < m_layers.size () - 1; ++i )
    {
        MathAI::Vector layer_output ( m_layers [ i + 1 ] );

        for ( size_t j = 0; j < m_layers [ i + 1 ]; ++j )
        {
            // Compute network function.

            double sum = m_bias_vectors [ i ][ j ];

            for ( size_t k = 0; k < m_layers [ i ]; ++k )
            {
                double x = activations.back () [ k ];
                double w = m_weight_vectors [ i ][ k * m_layers [ i + 1 ] + j ];

                sum += w * x;
            }

            // Select and compute activation function.

            switch ( m_activation_functions [ i ] )
            {
                case LINEAR:  layer_output [ j ] = MathAI::Linear  ( sum ); break;
                case SIGMOID: layer_output [ j ] = MathAI::Sigmoid ( sum ); break;
                case TANH:    layer_output [ j ] = MathAI::Tanh    ( sum ); break;
                case RELU:    layer_output [ j ] = MathAI::ReLU    ( sum ); break;
            }
        }

        activations.push_back ( layer_output );
    }

    return activations;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: PropagateBackward
//
// Description:
//
//   Performs the backward pass for a single sample, computing the error gradients (deltas) for every layer by
//   propagating the output error backwards and scaling by each activation function's derivative.
//
// Arguments:
//
//   activations (const MathAI::Matrix&):
//     The per-layer activations produced by the corresponding forward pass.
//
//   targets (const MathAI::Vector&):
//     The expected output values for the sample.
//
// Returns:
//
//   MathAI::Matrix:
//     The error gradient (delta) for every layer, used to accumulate the weight and bias gradients.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Matrix NeuralNetwork::PropagateBackward ( const MathAI::Matrix& activations, const MathAI::Vector& targets )
{
    MathAI::Matrix deltas ( m_layers.size () );

    // Calculate output layer delta.

    deltas.back ().resize ( m_layers.back () );

    for ( size_t i = 0; i < m_layers.back (); ++i )
    {
        double error      = activations.back () [ i ] - targets [ i ];
        double activation = activations.back () [ i ];

        switch ( m_activation_functions.back () )
        {
            case LINEAR:  deltas.back () [ i ] = error * MathAI::LinearDerivative  ( activation ); break;
            case SIGMOID: deltas.back () [ i ] = error * MathAI::SigmoidDerivative ( activation ); break;
            case TANH:    deltas.back () [ i ] = error * MathAI::TanhDerivative    ( activation ); break;
            case RELU:    deltas.back () [ i ] = error * MathAI::ReLUDerivative    ( activation ); break;
        }
    }

    // Calculate hidden layer deltas.

    for ( size_t i = m_layers.size () - 2; i > 0; --i )
    {
        deltas [ i ].resize ( m_layers [ i ] );

        for ( size_t j = 0; j < m_layers [ i ]; ++j )
        {
            double activation = activations [ i ][ j ];
            double delta      = 0.0;

            for ( size_t k = 0; k < m_layers [ i + 1 ]; ++k )
            {
                delta += deltas [ i + 1 ][ k ] * m_weight_vectors [ i ][ j * m_layers [ i + 1 ] + k ];
            }

            switch ( m_activation_functions [ i - 1 ] )
            {
                case LINEAR:  deltas [ i ][ j ] = delta * MathAI::LinearDerivative  ( activation ); break;
                case SIGMOID: deltas [ i ][ j ] = delta * MathAI::SigmoidDerivative ( activation ); break;
                case TANH:    deltas [ i ][ j ] = delta * MathAI::TanhDerivative    ( activation ); break;
                case RELU:    deltas [ i ][ j ] = delta * MathAI::ReLUDerivative    ( activation ); break;
            }
        }
    }

    return deltas;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ComputeLoss
//
// Description:
//
//   Computes the mean squared error of the network over an entire data set by running a forward pass for every
//   sample and averaging the squared output errors. Used to monitor training progress at the end of each epoch.
//
// Arguments:
//
//   training_data_x (const MathAI::Matrix&):
//     The input features of the data set.
//
//   training_data_y (const MathAI::Matrix&):
//     The corresponding target outputs.
//
// Returns:
//
//   double:
//     The mean squared error averaged over all samples.
//
//---------------------------------------------------------------------------------------------------------------------

double NeuralNetwork::ComputeLoss ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y )
{
    double training_data_sample_count = (double) training_data_x.size ();
    double loss                       = 0.0;
    double average_loss               = 0.0;

    // Accumulate loss measurements.

    for ( size_t i = 0; i < training_data_x.size (); ++i )
    {
        MathAI::Matrix        activations = PropagateForward ( training_data_x [ i ] );
        const MathAI::Vector& output      = activations.back ();

        for ( size_t j = 0; j < output.size (); ++j )
        {
            double error = output [ j ] - training_data_y [ i ][ j ];
            loss += error * error;
        }
    }

    // Compute average loss.

    average_loss = loss / training_data_sample_count;

    return average_loss;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Predict
//
// Description:
//
//   Performs regression inference. Runs a single forward pass and returns the activations of the output layer as a
//   continuous-valued vector.
//
// Arguments:
//
//   input (const MathAI::Vector&):
//     The feature vector to run through the network. Its length must match the input-layer size.
//
// Returns:
//
//   MathAI::Vector:
//     The output-layer activations -- the network's continuous prediction for the input.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Vector NeuralNetwork::Predict ( const MathAI::Vector& input )
{
    // Perform forward propagation on the input data.

    MathAI::Matrix activations = PropagateForward ( input );

    // The output of the network is the activations of the last layer.

    return activations.back ();
}

//---------------------------------------------------------------------------------------------------------------------
// Method: Classify
//
// Description:
//
//   Performs classification inference. Runs a single forward pass and returns the index of the output neuron with
//   the largest activation (the arg-max class).
//
// Arguments:
//
//   input (const MathAI::Vector&):
//     The feature vector to classify. Its length must match the input-layer size.
//
// Returns:
//
//   int:
//     The zero-based index of the highest-scoring output neuron.
//
//---------------------------------------------------------------------------------------------------------------------

int NeuralNetwork::Classify ( const MathAI::Vector& input )
{
    // Perform forward propagation on the input data.

    MathAI::Matrix activations = PropagateForward ( input );

    // The output of the network is the activations of the last layer.

    std::vector<double> output = activations.back ();

    // Find the index of the maximum value in the output vector.

    auto max_element_it =       max_element ( output.begin (), output.end () );
    int predicted_class = (int) distance    ( output.begin (), max_element_it );

    return predicted_class;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: SaveModel
//
// Description:
//
//   Saves the trained model (weights and biases) to a file. Not yet implemented -- currently a stub reserved for
//   future work.
//
// Arguments:
//
//   model_path (const string&):
//     Path of the file to which the model would be written.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::SaveModel ( const string& model_path )
{
    // Stub for saving model.
}

//---------------------------------------------------------------------------------------------------------------------
// Method: LoadModel
//
// Description:
//
//   Loads a previously saved model (weights and biases) from a file. Not yet implemented -- currently a stub
//   reserved for future work.
//
// Arguments:
//
//   model_path (const string&):
//     Path of the file from which the model would be read.
//
//---------------------------------------------------------------------------------------------------------------------

void NeuralNetwork::LoadModel ( const string& model_path )
{
    // Stub for loading model.
}

//---------------------------------------------------------------------------------------------------------------------
// Method: TrainingProgressToString
//
// Description:
//
//   Formats a single-line training-progress message showing the current epoch, the total epoch count, the
//   percentage complete, and the current loss. The line ends with a carriage return so successive updates
//   overwrite one another on the console.
//
// Arguments:
//
//   epoch_index (size_t):
//     The zero-based index of the current epoch.
//
//   epoch_count (size_t):
//     The total number of epochs.
//
//   loss (double):
//     The loss measured at the end of the current epoch.
//
// Returns:
//
//   string:
//     The formatted progress line, terminated with a carriage return.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::TrainingProgressToString ( size_t epoch_index, size_t epoch_count, double loss )
{
    // Initialise local constants.

    const string CARIAGE_RETURN               = "\r";
    const string TRAIN                        = "[Train] ";
    const string EPOCH_FIELD                  = "Epoch: ";
    const string OUT_OF                       = " / ";
    const string DELIMITER                    = ", ";
    const string LOSS_FIELD                   = "Loss: ";
    const string PARENTHESIS_PERCENTAGE_OPEN  = " (";
    const string PARENTHESIS_PERCENTAGE_CLOSE = " %)";

    // Initialise local variables.

    size_t training_progress_percentage        = ( size_t ) ( 100 * ( epoch_index + 1 ) / epoch_count );  // Compute training progress.
    string epoch_index_string                  = to_string ( epoch_index + 1 );
    string epoch_count_string                  = to_string ( epoch_count );
    string training_progress_percentage_string = to_string ( training_progress_percentage );
    string loss_string                         = to_string ( loss );
    string training_progress_string            = "";

    // Compile progress strings.

    training_progress_string += TRAIN;
    training_progress_string += EPOCH_FIELD + epoch_index_string + OUT_OF + epoch_count_string;
    training_progress_string += PARENTHESIS_PERCENTAGE_OPEN + training_progress_percentage_string + PARENTHESIS_PERCENTAGE_CLOSE;
    training_progress_string += DELIMITER + LOSS_FIELD + loss_string;
    training_progress_string += CARIAGE_RETURN;

    // Return progress string to caller.

    return training_progress_string;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ToString
//
// Description:
//
//   Builds a human-readable, multi-line summary of the network's architecture and hyperparameters: each layer's
//   type, neuron count, and activation function, plus the loss function, optimisation algorithm, learning rate,
//   epoch count, and batch size.
//
// Returns:
//
//   string:
//     A formatted, multi-line description of the network configuration.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::ToString () const
{
    // Local constants.

    const string NEW_LINE            = "\n";
    const string NA                  = "N/A";
    const string BULLET              = "- ";
    const string DELIMITER_1         = ", ";
    const string DELIMITER_2         = ": ";
    const string ASSIGNMENT          = " = ";
    const string LAYER_SIZE          = "Neuron Count";
    const string ACTIVATION_FUNCTION = "Activation Function";
    const string LAYER               = "Layer ";
    const string LAYER_INPUT         = "[ Input  ] ";
    const string LAYER_HIDDEN        = "[ Hidden ] ";
    const string LAYER_OUTPUT        = "[ Output ] ";

    // Initialise local variables.

    string bullet                    = "- ";
    string application_string        = "";
    size_t layer_count               = this->m_layers.size ();
    size_t activation_function_count = this->m_activation_functions.size ();

    // Compile string.

    application_string += NEW_LINE;
    application_string += "Neural Network Layers:" + NEW_LINE;

    // Compile string: Compile layer info.

    for ( size_t i = 0; i < layer_count; i++ )
    {
        string layer_info                 = "";
        string layer_size                 = to_string ( this->m_layers [ i ] );
        string layer_index_string         = to_string ( i );
        string activation_function_string = "";
        string layer_type                 = "";
        size_t activation_function_index = ( ( i - 1 >= 0 ) && ( i - 1 < activation_function_count ) ) ? i - 1 : 0;

        if ( i == 0 )
        {
            layer_type                 = LAYER_INPUT;
            activation_function_string = NA;
        }
        else if ( i < layer_count - 1 )
        {
            layer_type                 = LAYER_HIDDEN;
            activation_function_string = ActivationFunctionToString ( this->m_activation_functions [ activation_function_index ] );
        }
        else
        {
            layer_type                 = LAYER_OUTPUT;
            activation_function_string = ActivationFunctionToString ( this->m_activation_functions [ activation_function_index ] );
        }

        layer_info += bullet + layer_type + LAYER + layer_index_string + DELIMITER_2;
        layer_info += LAYER_SIZE + ASSIGNMENT + layer_size + DELIMITER_1;
        layer_info += ACTIVATION_FUNCTION + ASSIGNMENT + activation_function_string + NEW_LINE;

        application_string += layer_info;
    }

    // Compile scalar meta-parameters.

    application_string += NEW_LINE;
    application_string += "Loss Function          = " + LossFunctionToString          ( this->m_loss_function          ) + NEW_LINE;
    application_string += "Optimization Algorythm = " + OptimizationAlgorithmToString ( this->m_optimization_algorithm ) + NEW_LINE;
    application_string += "Learning Rate          = " + FormatFloat                   ( this->m_learning_rate          ) + NEW_LINE;
    application_string += "Epoch Count            = " + to_string                     ( this->m_epoch_count            ) + NEW_LINE;
    application_string += "Batch Count            = " + to_string                     ( this->m_batch_size             ) + NEW_LINE;

    return ( application_string );
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ActivationFunctionToString
//
// Description:
//
//   Returns the name of an activation-function enum value as a string, for display and logging.
//
// Arguments:
//
//   activation_function (ActivationFunction):
//     The activation-function value to convert.
//
// Returns:
//
//   string:
//     The name of the activation function (for example, "RELU"), or an empty string if not recognised.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::ActivationFunctionToString ( ActivationFunction activation_function ) const
{
    switch ( activation_function )
    {
        case ActivationFunction::LINEAR:  return "LINEAR";
        case ActivationFunction::SIGMOID: return "SIGMOID";
        case ActivationFunction::TANH:    return "TANH";
        case ActivationFunction::RELU:    return "RELU";
        default:                          return "";
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: LossFunctionToString
//
// Description:
//
//   Returns the name of a loss-function enum value as a string, for display and logging.
//
// Arguments:
//
//   loss_function (LossFunction):
//     The loss-function value to convert.
//
// Returns:
//
//   string:
//     The name of the loss function (for example, "MEAN_SQUARED_ERROR"), or an empty string if not recognised.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::LossFunctionToString ( LossFunction loss_function ) const
{
    switch ( loss_function )
    {
        case LossFunction::MEAN_SQUARED_ERROR: return "MEAN_SQUARED_ERROR";
        case LossFunction::CROSS_ENTROPY:      return "CROSS_ENTROPY";
        default:                               return "";
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: OptimizationAlgorithmToString
//
// Description:
//
//   Returns the name of an optimisation-algorithm enum value as a string, for display and logging.
//
// Arguments:
//
//   optimization_algorythm (OptimizationAlgorithm):
//     The optimisation-algorithm value to convert.
//
// Returns:
//
//   string:
//     The name of the optimisation algorithm (for example, "STOCHASTIC_GRADIENT_DESCENT"), or empty if not
//     recognised.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::OptimizationAlgorithmToString ( OptimizationAlgorithm optimization_algorythm ) const
{
    switch ( optimization_algorythm )
    {
        case OptimizationAlgorithm::GRADIENT_DESCENT:            return "GRADIENT_DESCENT";
        case OptimizationAlgorithm::STOCHASTIC_GRADIENT_DESCENT: return "STOCHASTIC_GRADIENT_DESCENT";
        case OptimizationAlgorithm::ADAM:                        return "ADAM";
        default:                                                 return "";
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: FormatFloat
//
// Description:
//
//   Formats a floating-point value as a string with any trailing zeros, and a trailing decimal point, removed,
//   giving a compact representation for display.
//
// Arguments:
//
//   value (double):
//     The value to format.
//
// Returns:
//
//   string:
//     The value rendered as a compact decimal string.
//
//---------------------------------------------------------------------------------------------------------------------

string NeuralNetwork::FormatFloat ( double value ) const
{
    string str = std::to_string ( value );

    str.erase ( str.find_last_not_of ( '0' ) + 1, string::npos );
    str.erase ( str.find_last_not_of ( '.' ) + 1, string::npos );

    return str;
}


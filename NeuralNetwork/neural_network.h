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
//---------------------------------------------------------------------------------------------------------------------

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>

#include "math_ai.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------------------
// Enum: ActivationFunction
//
// Description:
//
//   Enumerates the per-layer activation functions supported by the network. LINEAR is the identity function, used for
//   regression outputs; SIGMOID and TANH are the classic saturating activations; RELU is the rectified linear unit
//   used by default for hidden layers.
//
//---------------------------------------------------------------------------------------------------------------------

enum ActivationFunction    { LINEAR, SIGMOID, TANH, RELU };

//---------------------------------------------------------------------------------------------------------------------
// Enum: LossFunction
//
// Description:
//
//   Enumerates the loss (cost) functions the network can be configured with. MEAN_SQUARED_ERROR is used for regression
//   and is the implemented path; CROSS_ENTROPY is reserved for classification and is not yet implemented.
//
//---------------------------------------------------------------------------------------------------------------------

enum LossFunction          { MEAN_SQUARED_ERROR, CROSS_ENTROPY };

//---------------------------------------------------------------------------------------------------------------------
// Enum: OptimizationAlgorithm
//
// Description:
//
//   Enumerates the weight-update strategies the network can be configured with. GRADIENT_DESCENT and
//   STOCHASTIC_GRADIENT_DESCENT drive the implemented mini-batch training loop; ADAM is reserved for future work and
//   is not yet implemented.
//
//---------------------------------------------------------------------------------------------------------------------

enum OptimizationAlgorithm { GRADIENT_DESCENT, STOCHASTIC_GRADIENT_DESCENT, ADAM };

//*********************************************************************************************************************
// Class: NeuralNetwork
//
// Description:
//
//   A general-purpose, fully connected feed-forward neural network (multi-layer perceptron) supporting both regression
//   and classification. The architecture -- layer sizes, per-layer activation functions, loss function, optimisation
//   algorithm, learning rate, batch size, and epoch count -- is fixed at construction. Train fits the weights and
//   biases by backpropagation and gradient descent; Predict performs regression inference and Classify performs
//   classification inference. Per-epoch loss is logged to a CSV file during training.
//
//*********************************************************************************************************************

class NeuralNetwork
{
public:

    // Public data types.

    //typedef vector <double>          Vector;
    //typedef vector <vector <double>> Matrix;

    //=================================================================================================================
    // Constructors
    //=================================================================================================================

    NeuralNetwork
    (
        const vector <int>&                layers,
        const vector <ActivationFunction>& activation_functions,
        LossFunction                       loss_function,
        double                             learning_rate,
        int                                epoch_count,
        int                                batch_size,
        OptimizationAlgorithm              optimization_algorithm,
        const string                       training_results_file_name
    );

    //=================================================================================================================
    // Methods
    //=================================================================================================================

    void           Train                         ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y );
    MathAI::Vector Predict                       ( const MathAI::Vector& input );
    int            Classify                      ( const MathAI::Vector& input );
    string         ToString                      () const;
    string         ActivationFunctionToString    ( ActivationFunction    activation_function    ) const;
    string         LossFunctionToString          ( LossFunction          loss_function          ) const;
    string         OptimizationAlgorithmToString ( OptimizationAlgorithm optimization_algorythm ) const;
    void           SaveModel                     ( const string& model_path );
    void           LoadModel                     ( const string& model_path );
    string         FormatFloat                   ( double value ) const;

private:

    //=================================================================================================================
    // Methods
    //=================================================================================================================

    void           InitializeWeights        ();
    MathAI::Matrix PropagateForward         ( const MathAI::Vector& input );
    MathAI::Matrix PropagateBackward        ( const MathAI::Matrix& activations, const MathAI::Vector& targets );
    double         ComputeLoss              ( const MathAI::Matrix& training_data_x, const MathAI::Matrix& training_data_y );
    void           InitializeGradients      ( MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients );
    void           AccumulateGradients      ( const MathAI::Matrix& activations, const MathAI::Matrix& error_gradients, MathAI::Matrix& weight_gradients, MathAI::Matrix& bias_gradients );
    void           UpdateWeightsAndBiases   ( const MathAI::Matrix& weight_gradients, const MathAI::Matrix& bias_gradients );
    string         TrainingProgressToString ( size_t epoch_index, size_t epoch_count, double loss );
    void           ShowCursor               ( bool cursor_hidden );

    //=================================================================================================================
    // Data Members
    //=================================================================================================================

    vector <int>                m_layers;
    vector <ActivationFunction> m_activation_functions;
    LossFunction                m_loss_function;
    double                      m_learning_rate;
    int                         m_epoch_count;
    int                         m_batch_size;                   // The number of training samples per batch.
    OptimizationAlgorithm       m_optimization_algorithm;
    MathAI::Matrix              m_weight_vectors;
    MathAI::Matrix              m_bias_vectors;
    string                      m_training_results_file_name;   // CSV file listing the results of a taining run.
};

#endif // NEURAL_NETWORK_H

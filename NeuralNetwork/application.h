//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 5.1
// Date:    2014
// Author:  Rohin Gosling
//
// Description:
//
//   Declaration of the Application class, the demonstration harness for the Common Artificial Intelligence Library.
//   The class drives a complete end-to-end example: it generates synthetic training data, constructs and trains a
//   NeuralNetwork, and then runs an interactive console loop that performs inference on user-supplied inputs. It also
//   provides the CSV-loading and table-manipulation helpers used to prepare the training data.
//
//---------------------------------------------------------------------------------------------------------------------

#ifndef APPLICATION_H
#define APPLICATION_H

#include <string>
#include <vector>

#include "neural_network.h"
#include "math_ai.h"

using namespace std;

//*********************************************************************************************************************
// Class: Application
//
// Description:
//
//   Top-level driver for the demonstration program. A single Application instance owns the program workflow exposed
//   through Run: generating a synthetic data set, building and training a NeuralNetwork, and entering an interactive
//   inference loop. Supporting methods load CSV data into a MathAI::Matrix, split a table into feature and target
//   sub-matrices, generate test data, and format the object's state for display.
//
//*********************************************************************************************************************

class Application
{
public:

    //=================================================================================================================
    // Constructors
    //=================================================================================================================

    Application ();

    //=================================================================================================================
    // Methods
    //=================================================================================================================

    void           Run               ();
    string         ToString          () const;
    MathAI::Matrix LoadTrainingData  ( const string& filename, bool ignore_header_row );
    MathAI::Matrix SplitTable        ( const MathAI::Matrix& matrix, long column_index_first, long column_index_last );
    void           GenerateTestData  ( const string& file_name, long row_count );
    void           TestTerminal      ( NeuralNetwork& neural_network );

    //=================================================================================================================
    // Constants
    //=================================================================================================================

    const string TERMINAL_MESSAGE_APPLICATION = "[Application] ";
    const string TERMINAL_MESSAGE_SYSTEM      = "[SYSTEM] ";
    const string TERMINAL_MESSAGE_EXCEPTION   = "[EXCEPTION] ";
    const string TERMINAL_MESSAGE_ERROR       = "[ERROR] ";

private:

    //=================================================================================================================
    // Data Members
    //=================================================================================================================

    string application_name;
    string application_version;
};

#endif // APPLICATION_H

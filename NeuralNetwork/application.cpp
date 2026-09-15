//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 5.1
// Date:    2014
// Author:  Rohin Gosling
//
// Description:
//
//   Implementation of the Application class, the demonstration harness for the Common Artificial Intelligence Library.
//   Run drives the end-to-end workflow -- synthetic data generation, neural network construction and training, and an
//   interactive inference loop -- while the remaining methods provide the supporting CSV loading, table splitting,
//   test-data generation, and state-formatting routines.
//
// TODO:
//
//   1. Add model save and load functions to NeuralNetwork.
//
//---------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <random>
#include <vector>
#include <cmath>

#include "application.h"
#include "neural_network.h"
#include "math_ai.h"

using namespace std;

//=====================================================================================================================
// Constructors
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
// Constructor 1/1: Application
//
// Description:
//
//   Constructs the Application and initialises its descriptive metadata -- the application name ("Neural Network Test
//   Application") and version string reported by ToString. No neural network or data is created at construction time;
//   all work is performed when Run is called.
//
//---------------------------------------------------------------------------------------------------------------------

Application::Application ()
{
    this->application_name    = "Neural Network Test Application";
    this->application_version = "5.1";
}

//=====================================================================================================================
// Methods
//=====================================================================================================================

//---------------------------------------------------------------------------------------------------------------------
// Method: Run
//
// Description:
//
//   Executes the full demonstration workflow. Generates a synthetic training-data CSV, constructs and configures a
//   NeuralNetwork, loads the data and splits it into feature and target matrices, trains the network, and finally
//   hands over to the interactive terminal test loop. Any exceptions raised during the run are caught and reported to
//   the console.
//
//---------------------------------------------------------------------------------------------------------------------

void Application::Run ()
{
    // Initialise local varaibles.

    bool           training_data_generate_enabled       = true;
    bool           training_data_file_ignore_header_row = true;
    string         training_data_file_name              = "Data\\training_data.csv";
    string         training_results_file_name           = "Data\\training_results.csv";
    long           training_data_row_count              = 10000;
    long           training_data_feature_count          = 2;
    long           training_data_target_count           = 1;
    MathAI::Matrix training_data;
    MathAI::Matrix training_data_x;
    MathAI::Matrix training_data_y;

    try
    {
        // Display program information.

        cout << ToString ();

        // Generate test data.

        if ( training_data_generate_enabled )
        {
            GenerateTestData ( training_data_file_name, training_data_row_count );

            cout << endl << TERMINAL_MESSAGE_APPLICATION + "Training data generated and saved to file, \"" + training_data_file_name + "\"\n";
        }

        // Initialise neural network.

        vector <int>                layers                 = { 2, 3, 1 };
        vector <ActivationFunction> activation_functions   = { RELU, LINEAR };
        LossFunction                loss_function          = MEAN_SQUARED_ERROR;
        double                      learning_rate          = 0.001;
        int                         epoch_count            = 200;
        int                         batch_size             = 50;
        OptimizationAlgorithm       optimization_algorithm = STOCHASTIC_GRADIENT_DESCENT;

        NeuralNetwork neural_network
        (
            layers,
            activation_functions,
            loss_function,
            learning_rate,
            epoch_count,
            batch_size,
            optimization_algorithm,
            training_results_file_name
        );

        cout << endl << TERMINAL_MESSAGE_APPLICATION + "Neural network initialised...\n" + neural_network.ToString ();

        // Load training data.
        // - CSV file is loaded into a 2D array of doubles, `vector <vector <double>>`.
        // - For our training data in this example, columns 0 to 2 are the input features of our training data, and column 3 is the targets.

        training_data   = LoadTrainingData ( training_data_file_name, training_data_file_ignore_header_row );
        training_data_x = SplitTable       ( training_data, 0,                           training_data_feature_count - 1                              );
        training_data_y = SplitTable       ( training_data, training_data_feature_count, training_data_feature_count + training_data_target_count - 1 );

        // Train the neural nwtork.

        cout << endl << TERMINAL_MESSAGE_APPLICATION + "Training neural network..." << endl << endl;

        neural_network.Train ( training_data_x, training_data_y );

        //nn.SaveModel("model.csv");
        //nn.LoadModel("model.csv");

        // Test neural network.

        TestTerminal ( neural_network );
    }
    catch ( const runtime_error& e )
    {
        // Handle runtime_error exceptions.

        cerr << endl << TERMINAL_MESSAGE_ERROR << e.what () << endl;
    }
    catch ( const exception& e )
    {
        // Handle all other standard exceptions.

        cerr << endl << TERMINAL_MESSAGE_EXCEPTION << e.what () << endl;
    }
    catch ( ... )
    {
        // Handle any other exceptions.

        cerr << endl << TERMINAL_MESSAGE_ERROR << "An unknown error occurred." << endl;
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: TestTerminal
//
// Description:
//
//   Runs an interactive inference loop on the console. Repeatedly reads two integer inputs from the user, encodes them
//   into the network's [0, 1] input range, queries the trained network for a prediction, decodes the result back to
//   the original scale, and prints it. The loop continues until the user enters "exit".
//
// Arguments:
//
//   neural_network (NeuralNetwork&):
//     The trained network used to perform inference on the user-supplied inputs.
//
//---------------------------------------------------------------------------------------------------------------------

void Application::TestTerminal ( NeuralNetwork& neural_network )
{
    // Example inference

    string input;
    int    x0i;
    int    x1i;
    double data_scale      = 1000.0;
    int    test_case_index = 1;

    while ( true )
    {
        // Get test values from terminal and convert input to integer.

        cout << endl << "------------------------------------------------[ " << test_case_index << " ]---" << endl;

        cout << "Enter Value x0: "; getline ( cin, input ); if ( input == "exit" ) break;
        istringstream x0s ( input ); x0s >> x0i;

        cout << "Enter Value x1: "; getline ( cin, input ); if ( input == "exit" ) break;
        istringstream x1s ( input ); x1s >> x1i;

        // Encode inputs.

        double x0 = ( double ) x0i / data_scale;
        double x1 = ( double ) x1i / data_scale;

        // Test neural network.

        MathAI::Vector x = { x0, x1 };
        MathAI::Vector y = neural_network.Predict ( x );

        //  Decode output.

        double y0 = ( y [ 0 ] * data_scale * 2.0 );

        // Print results.

        cout << endl << TERMINAL_MESSAGE_APPLICATION << "x  = { "; for ( int i = 0; i < x.size (); ++i ) if ( i == x.size () - 1 ) cout << x [ i ]; else cout << x [ i ] << ", "; cout << " }";
        cout << endl << TERMINAL_MESSAGE_APPLICATION << "y0 = " << y [ 0 ];
        cout << endl << TERMINAL_MESSAGE_APPLICATION << "y  = " << y0 << endl;

        // Incrament test case index.

        test_case_index++;
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Method: SplitTable
//
// Description:
//
//   Extracts a contiguous range of columns from a table, returning a new matrix that contains only the selected
//   columns of every row. Used by Run to separate the loaded data set into its input-feature and target sub-matrices.
//
// Arguments:
//
//   matrix (const MathAI::Matrix&):
//     The source table to slice. Every row must have at least column_index_last + 1 columns.
//
//   column_index_first (long):
//     Index of the first column to copy, inclusive and zero-based.
//
//   column_index_last (long):
//     Index of the last column to copy, inclusive and zero-based.
//
// Returns:
//
//   MathAI::Matrix:
//     A matrix containing columns column_index_first through column_index_last of each input row. Throws
//     std::out_of_range if the column range is invalid.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Matrix Application::SplitTable ( const MathAI::Matrix& matrix, long column_index_first, long column_index_last )
{
    // Initialize result.

    MathAI::Matrix result;

    // Check for valid column range.

    if ( column_index_first < 0 || column_index_last >= matrix [ 0 ].size () || column_index_first > column_index_last )
    {
        throw out_of_range ( "SplitTable 1/2: Invalid column index range." );
    }
    else
    {
        // Iterate through each row of the original table.

        for ( const auto& row : matrix )
        {
            MathAI::Vector new_row;

            for ( long i = column_index_first; i <= column_index_last; ++i )
            {
                new_row.push_back ( row [ i ] );
            }

            result.push_back ( new_row );
        }
    }

    return result;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: LoadTrainingData
//
// Description:
//
//   Loads a CSV file into a MathAI::Matrix, parsing each comma-separated field as a double. One matrix row is produced
//   per CSV line, optionally skipping the first (header) row.
//
// Arguments:
//
//   filename (const string&):
//     Path to the CSV file to read.
//
//   ignore_header_row (bool):
//     When true, the first line of the file is treated as a header and skipped; when false, it is parsed as data.
//
// Returns:
//
//   MathAI::Matrix:
//     A row-major matrix of the parsed values, where each inner vector is one CSV row.
//
//---------------------------------------------------------------------------------------------------------------------

MathAI::Matrix Application::LoadTrainingData ( const string& filename, bool ignore_header_row )
{
    // Initialise local varaibles.

    MathAI::Matrix data;
    char           csv_file_delimiter = ',';
    ifstream       csv_file ( filename );
    string         csv_file_line;
    long           csv_file_line_index;

    // Load training data

    csv_file_line_index = 0;

    while ( getline ( csv_file, csv_file_line ) )
    {
        // If `ignore_header_row` is true, we will skip the header row. Otherwise we'll add it.

        if ( ( csv_file_line_index > 0 ) || ( csv_file_line_index == 0 && !ignore_header_row ) )
        {
            stringstream   csv_file_line_string ( csv_file_line );
            MathAI::Vector data_row;
            string         data_value;

            // Get a row of data from the CSV file, and add to the data vector.

            while ( getline ( csv_file_line_string, data_value, csv_file_delimiter ) )
            {
                data_row.push_back ( stod ( data_value ) );
            }

            data.push_back ( data_row );
        }

        csv_file_line_index++;
    }

    // Return loadded data.

    return data;
}

//---------------------------------------------------------------------------------------------------------------------
// Method: GenerateTestData
//
// Description:
//
//   Generates a synthetic regression data set and writes it to a CSV file. Each row holds two random features in the
//   interval [0, 1] and a target equal to their normalised sum, producing the data on which the demo network learns to
//   add two numbers.
//
// Arguments:
//
//   file_name (const string&):
//     Path of the CSV file to create. Any existing file is overwritten.
//
//   row_count (long):
//     The number of data rows to generate, excluding the header row.
//
//---------------------------------------------------------------------------------------------------------------------

void Application::GenerateTestData ( const string& file_name, long row_count )
{
    ofstream outFile ( file_name );

    if ( !outFile.is_open () )
    {
        cerr << "Error: Unable to open file for writing." << endl;
        return;
    }

    // Initialise local variables.

    double data_scale          = 1000;
    double normalization_scale = 1000;

    // Write the header row.

    outFile << "feature_x0,feature_x1,target_y\n";

    // Set up the random number generator.

    random_device                rd;
    mt19937                      gen ( rd() );
    uniform_real_distribution <> dis ( 0.0, 1.0 );

    // Generate data rows.

    for ( int i = 0; i < row_count; ++i )
    {
        // Generate test data.

        double feature_x0 = round ( dis ( gen ) * data_scale );
        double feature_x1 = round ( dis ( gen ) * data_scale );
        double target_y   = feature_x0 + feature_x1;

        // Encode test data.

        double feature_x0_encoded = feature_x0 / normalization_scale;
        double feature_x1_encoded = feature_x1 / normalization_scale;
        double target_y_encoded   = ( target_y / 2.0 ) / normalization_scale;

        // Write the data row to the CSV file.

        outFile << fixed << setprecision ( (streamsize) 3 )
            << feature_x0_encoded << ","
            << feature_x1_encoded << ","
            << target_y_encoded   << "\n";
    }

    outFile.close();
}

//---------------------------------------------------------------------------------------------------------------------
// Method: ToString
//
// Description:
//
//   Builds a human-readable summary of the application's metadata, listing the application name and version on
//   separate lines.
//
// Returns:
//
//   string:
//     A multi-line string containing the application name and version.
//
//---------------------------------------------------------------------------------------------------------------------

string Application::ToString () const
{
    string application_string = "";

    application_string += "Application Name : " + this->application_name    + "\n";
    application_string += "Version          : " + this->application_version + "\n";

    return ( application_string );
}

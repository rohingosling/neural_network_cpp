//---------------------------------------------------------------------------------------------------------------------
// Project: Common Artificial Intelligence Library
// Version: 5.1
// Date:    2014
// Author:  Rohin Gosling
//
// Description:
//
//   Application entry point for the Common Artificial Intelligence Library demonstration program. This translation
//   unit contains main, which constructs the Application object and hands control to it. All program behaviour --
//   synthetic training data generation, neural network construction and training, and the interactive inference loop
//   -- is driven by Application::Run, keeping the entry point itself deliberately minimal.
//
//---------------------------------------------------------------------------------------------------------------------

#include <iostream>
#include "application.h"

using namespace std;

//---------------------------------------------------------------------------------------------------------------------
// Method: main
//
// Description:
//
//   Program entry point. Instantiates a single Application object on the stack and delegates the entire program
//   workflow to Application::Run, returning a success status code to the host operating system once Run completes.
//
// Returns:
//
//   int:
//     Process exit code returned to the host operating system. A value of 0 indicates successful execution.
//
//---------------------------------------------------------------------------------------------------------------------

int main()
{
    Application application;
    application.Run();

    return 0;
}

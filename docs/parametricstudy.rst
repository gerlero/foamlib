Parametric Study
================

This functionality allows users to set up and run a parametric study in OpenFOAM, where multiple simulations are executed with varying parameters to analyze their effects on the results. The study can be configured to modify specific fields or solver settings across different runs.

Overview
--------

The general workflow for a parametric study in OpenFOAM using foamlib is as follows:

.. mermaid::

   graph LR
     A[Template Case] --> B[Generate Case Variants]
     C[Modify Parameters] --> B
     B --> D[Run Simulation]
     D --> E[Post-process and Analyze]

A template case is created with the necessary configuration files. The user can then generate multiple case variants by modifying specific parameters, such as initial conditions, boundary conditions, or solver settings. Each variant is run independently, and the results are collected for post-processing and analysis.



The general concept of the parametric study is that template cases are copied to a new folder, and the parameters are modified in the copied case. This requires the definition of the following parameters:

- template_case: The path to the template case that will be copied and modified.
- output_folder: The folder where the modified cases will be stored.
- case_name: The name of the case that will be created.
- instructions: A list of the file and the key entry that needs to be modified in the template case.
- value: the value for each instruction
- case category: A category for the case, which can be used to group cases together for easier post-processing.


Multiple parammetric study generators are avaible and describe below in detail.

CSV Generator
~~~~~~~~~~~~~

The csv generator create the parametric study based on a CSV file that contains the parameters to be varied. This generator reads the CSV file, extracts the parameters, and generates multiple case variants by modifying the specified fields in the template case.

.. code-block:: python

    from foamlib.preprocessing.parameter_study import csv_generator

    # Example usage
    csv_generator(
        csv_file="path/to/your/parameters.csv",
        template_case="path/to/template/case",
        output_folder="path/to/output/folder"
    ).create_study()

This simple generator specifies the above requirements in the csv file, where the instruction (file and key name (here: NX,NY and someModel)) is defined in the `system/simulationsParameters` file. The case_name  and the category will be defined as additional columns in the CSV file.


.. code-block:: c++

    /*--------------------------------*- C++ -*----------------------------------*\
    | =========                 |                                                 |
    | \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
    |  \\    /   O peration     | Version:  plus                                  |
    |   \\  /    A nd           | Web:      www.OpenFOAM.com                      |
    |    \\/     M anipulation  |                                                 |
    \*---------------------------------------------------------------------------*/
    FoamFile
    {
        version     2.0;
        format      ascii;
        class       dictionary;
        location    "system";
        object      simulationsParameters;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    NX 23;
    NY 8;

    someModel SomeModelName;


    // ************************************************************************* //

The file can be included in every openfoam dictionary by adding the following line and can be referenced with a dollar sign in the dictionary:

.. code-block:: c++

    #include "system/simulationsParameters"

    blocks
    (
        hex (0 1 2 3 4 5 6 7) ($NX $NY 1) simpleGrading (1 1 1)
    );

    
The csv file needs to be contain a case_name column and the parameters to be varied. Additionally, columns can be specified to categorize the cases: `Resolution`, and `Model`. The generator will create a case for each row in the CSV file. This is necessary to simplify the post-processing. The csv file should look like this:


==========  ====  ====  ==========  ============  ===================
case_name   NX    NY    someModel   Resolution    Model
==========  ====  ====  ==========  ============  ===================
case_001    100   200   modelA      coarse        k-epsilon
case_002    150   300   modelB      fine          Spalart-Allmaras
==========  ====  ====  ==========  ============  ===================




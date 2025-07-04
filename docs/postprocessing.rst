Post Processing
===============

Analyzing large parametric studies can be cumbersome, especially when dealing with numerous cases and parameters. To facilitate this process, FoamLib provides a post-processing module that allows users to analyze and visualize the results of their parametric studies efficiently.

The general concept is that the post processing case are all located in the same folder and with the following structure:

- cases
   * case1
      + postProcessing
         - function object 1
         - function object 2
   * case2
      + postProcessing
         - function object 1
         - function object 2

The post-processing module can be used to extract data from the post-processing files and visualize it in a user-friendly manner. 

The main idea is to gather all the post-processing files from different cases into a single dataframe that stores the data in a long format. This allows for easy manipulation and visualization of the data using libraries like seaborn, ploty.express or altair or numerous other plotting libraries.

time series data
----------------


.. code-block:: python

    from foamlib.postprocessing.load_tables import OutputFile, list_outputfiles, load_tables

    forces = load_tables(
        source=OutputFile(file_name="force.dat", folder="forces"), dir_name="Cases"
    )
    forces.to_csv(
        results / "forces.csv",
        index=False,
    )

The following example would load all the force.dat files from the post-processing folder of each case in the Cases directory and save the results in a CSV file. The resulting dataframe will have columns for the case name, time, and force components (fx, fy, fz, ..) and the case category specified in the case.json file that gets automatically generated when creating a case with FoamLib pre-processing module.

The OutputFile class is used to specify the file name and folder where the post-processing files are located, where the general syntax in OpenFOAM is as followed:

.. code-block:: 

    postProcessing / folder1 / folder2 / timeName / file_name

Only the folder and file_name are required, the timeName is optional and can be used to specify a specific time folder. If not specified, the post-processing module will look for the file in all time folders.

list outputfiles
----------------

The `list_outputfiles` function can be used to list all the output files in a given directory, and the `load_tables` function can be used to load the data from the output files into a dataframe. The resulting dataframe will have columns for the case name, time, and force components (fx, fy, fz, ..) and the case category specified in the case.json file that gets automatically generated when creating a case with FoamLib pre-processing module.


.. code-block:: python

    out_files = list_outputfiles(root / "Cases")

    forces = load_tables(
        source=out_files["forces--force.dat"], dir_name=root / "Cases"
    )

spatial data (surfaces, sets, ...)
----------------------------------


The post-processing module also supports loading spatial data from OpenFOAM cases, such as surface data or sets. OpenFOAM stores for each time name a new file with the same name that contains the spatial data. These data can be loaded into a dataframe using the `load_tables` function, which will automatically handle the parsing of the data and return it in a long format.

However, the resulting dataframe may contain a lot of data, so these dataframe can be filtererd with a custum function that return a filtered dataframe. 

.. code-block:: python

    from foamlib.postprocessing.load_tables import OutputFile, load_tables

    def max_height_filter(table: pd.DataFrame, parameters: dict[str, str]) -> pd.DataFrame:
        """Filter the table to get the maximum height."""
        d = {
            "x": [table["x"].max()],
            "y": [table["y"].max()],
            "z": [table["z"].max()],
        }
        d.update(parameters)
        return pd.DataFrame(d)


    file = OutputFile(file_name="U_freeSurface.raw", folder="freeSurface")
    surface_heights = load_tables(
        source=file, dir_name=root / "Cases", filter_table=max_height_filter
    )
    surface_heights.to_csv(
        results / "surface_heights.csv",
        index=False,
    )

generally,the `load_tables` functions should be stored in a seperate file and the resulting tables should be written to disc as e.g csv feater or formats. The post-processing module can then be used to load the data from the files and visualize it in a user-friendly manner.

This allows the implementation of dashboards to quickly explore the data. Additionally, the table gathering process can be easily outgenerate from the CLI.
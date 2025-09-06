Quick Start Guide
=================

This guide will get you up and running with foamlib in just a few minutes.

Prerequisites
-------------

Before starting, make sure you have:

* Python 3.8+ installed
* foamlib installed (see :doc:`installation`)
* OpenFOAM installed (for running simulations)

Core Concepts
-------------

foamlib provides two main types of functionality:

1. **File Manipulation**: Read, write, and modify OpenFOAM files
2. **Case Management**: Clone, configure, run, and analyze OpenFOAM cases

File Manipulation
-----------------

Reading OpenFOAM Files
~~~~~~~~~~~~~~~~~~~~~~

foamlib can read any OpenFOAM file format:

.. code-block:: python

   from foamlib import FoamFile
   
   # Read a dictionary file
   control_dict = FoamFile("system/controlDict")
   print(f"End time: {control_dict['endTime']}")
   
   # Read field files
   from foamlib import FoamFieldFile
   
   velocity = FoamFieldFile("0/U")
   print(f"Velocity shape: {velocity.internal_field.shape}")

Writing OpenFOAM Files
~~~~~~~~~~~~~~~~~~~~~~

Create and modify OpenFOAM files programmatically:

.. code-block:: python

   from foamlib import FoamFile
   
   # Create a new dictionary
   properties = FoamFile()
   properties["version"] = "2.0"
   properties["format"] = "ascii"
   properties["class"] = "dictionary"
   properties["object"] = "transportProperties"
   
   # Add transport properties
   properties["nu"] = [0, 2, -1, 0, 0, 0, 0, 1e-06]  # Kinematic viscosity
   
   # Save to file
   properties.save("constant/transportProperties")

Working with Dimensions
~~~~~~~~~~~~~~~~~~~~~~~

foamlib provides convenient classes for handling OpenFOAM dimensions:

.. code-block:: python

   from foamlib import Dimensioned, DimensionSet
   
   # Create a dimensioned scalar
   viscosity = Dimensioned(
       "nu",
       DimensionSet(0, 2, -1, 0, 0, 0, 0),  # [m^2/s]
       1e-6
   )
   
   # Use in dictionary
   properties["nu"] = viscosity

Case Management
---------------

Cloning Tutorial Cases
~~~~~~~~~~~~~~~~~~~~~~

Start with OpenFOAM tutorial cases:

.. code-block:: python

   from foamlib import FoamCase
   
   # Clone a tutorial case
   case = FoamCase.clone_tutorial(
       "incompressible/simpleFoam/pitzDaily",
       destination="my_pitz_daily"
   )
   
   print(f"Case cloned to: {case.path}")

Running Cases
~~~~~~~~~~~~~

Execute simulations directly from Python:

.. code-block:: python

   # Run the case
   case.run()
   
   # Or run specific applications
   case.run(["blockMesh"])  # Generate mesh
   case.run(["simpleFoam"])  # Run solver

Configuring Cases
~~~~~~~~~~~~~~~~~

Modify case settings before running:

.. code-block:: python

   # Modify control dictionary
   case.control_dict["endTime"] = 1000
   case.control_dict["writeInterval"] = 100
   
   # Modify transport properties
   case["constant/transportProperties"]["nu"] = [0, 2, -1, 0, 0, 0, 0, 1.5e-05]
   
   # Save changes
   case.save()

Accessing Results
~~~~~~~~~~~~~~~~~

Extract and analyze simulation results:

.. code-block:: python

   # Get latest time directory
   latest_time = case.latest_time
   print(f"Latest time: {latest_time}")
   
   # Access field data
   velocity = case[latest_time]["U"]
   pressure = case[latest_time]["p"]
   
   # Work with data (NumPy arrays)
   import numpy as np
   print(f"Max velocity magnitude: {np.linalg.norm(velocity.internal_field, axis=1).max()}")
   print(f"Average pressure: {pressure.internal_field.mean()}")

Asynchronous Operations
-----------------------

For running multiple cases concurrently:

.. code-block:: python

   import asyncio
   from foamlib import AsyncFoamCase
   
   async def run_parametric_study():
       # Create multiple cases
       cases = []
       for i, viscosity in enumerate([1e-6, 1.5e-6, 2e-6]):
           case = AsyncFoamCase.clone_tutorial(
               "incompressible/simpleFoam/pitzDaily",
               destination=f"case_{i}"
           )
           # Set different viscosity
           case["constant/transportProperties"]["nu"] = [0, 2, -1, 0, 0, 0, 0, viscosity]
           case.save()
           cases.append(case)
       
       # Run all cases concurrently
       await asyncio.gather(*[case.run() for case in cases])
       
       # Analyze results
       for i, case in enumerate(cases):
           velocity = case[case.latest_time]["U"]
           max_vel = np.linalg.norm(velocity.internal_field, axis=1).max()
           print(f"Case {i}: Max velocity = {max_vel:.3f} m/s")
   
   # Run the study
   asyncio.run(run_parametric_study())

HPC Integration
---------------

For Slurm-based HPC clusters:

.. code-block:: python

   from foamlib import AsyncSlurmFoamCase
   
   # Create case for HPC submission
   case = AsyncSlurmFoamCase.clone_tutorial(
       "incompressible/simpleFoam/pitzDaily",
       destination="hpc_case"
   )
   
   # Configure job settings
   case.slurm_options = {
       "partition": "compute",
       "nodes": 1,
       "ntasks": 8,
       "time": "02:00:00"
   }
   
   # Submit and wait for completion
   await case.run()

Complete Example
----------------

Here's a complete example that demonstrates the main features:

.. code-block:: python

   from foamlib import FoamCase
   import numpy as np
   
   def analyze_pitz_daily():
       # Clone and setup case
       case = FoamCase.clone_tutorial(
           "incompressible/simpleFoam/pitzDaily",
           destination="my_analysis"
       )
       
       # Modify settings
       case.control_dict["endTime"] = 500
       case.control_dict["writeInterval"] = 50
       
       # Change viscosity
       case["constant/transportProperties"]["nu"] = [0, 2, -1, 0, 0, 0, 0, 1.5e-05]
       
       # Save and run
       case.save()
       case.run()
       
       # Analyze convergence
       forces_file = case.path / "postProcessing" / "forces" / "0" / "forces.dat"
       if forces_file.exists():
           print("Forces data found - case completed successfully!")
       
       # Extract final velocity field
       final_time = case.latest_time
       velocity = case[final_time]["U"]
       
       # Calculate statistics
       vel_magnitude = np.linalg.norm(velocity.internal_field, axis=1)
       print(f"Velocity statistics:")
       print(f"  Max: {vel_magnitude.max():.3f} m/s")
       print(f"  Mean: {vel_magnitude.mean():.3f} m/s")
       print(f"  Min: {vel_magnitude.min():.3f} m/s")
       
       return case
   
   # Run the analysis
   if __name__ == "__main__":
       case = analyze_pitz_daily()
       print(f"Analysis complete. Case location: {case.path}")

Next Steps
----------

Now that you've learned the basics, explore these topics:

* :doc:`examples/index` - More detailed examples and use cases
* :doc:`parametricstudy` - Setting up parametric studies
* :doc:`postprocessing` - Advanced post-processing techniques
* :doc:`api/index` - Complete API reference

Common Patterns
---------------

Here are some common patterns you'll use frequently:

**Case Comparison**

.. code-block:: python

   # Compare results between cases
   case1 = FoamCase("case1")
   case2 = FoamCase("case2")
   
   u1 = case1[case1.latest_time]["U"]
   u2 = case2[case2.latest_time]["U"]
   
   # Calculate difference
   diff = np.linalg.norm(u1.internal_field - u2.internal_field, axis=1)
   print(f"Max difference: {diff.max()}")

**Batch Processing**

.. code-block:: python

   # Process multiple time directories
   for time_dir in case.times:
       velocity = case[time_dir]["U"]
       # Process velocity field
       process_velocity_field(velocity)

**Configuration Templates**

.. code-block:: python

   def setup_case_template(case, reynolds_number):
       """Configure a case for given Reynolds number."""
       # Calculate viscosity for desired Re
       U_ref = 1.0  # Reference velocity
       L_ref = 0.1  # Reference length
       nu = U_ref * L_ref / reynolds_number
       
       # Set transport properties
       case["constant/transportProperties"]["nu"] = [0, 2, -1, 0, 0, 0, 0, nu]
       
       # Configure time stepping
       case.control_dict["deltaT"] = 0.01 / reynolds_number
       case.save()
       
       return case

This should give you a solid foundation for using foamlib effectively!
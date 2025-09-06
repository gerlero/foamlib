Case Management
===============

foamlib provides powerful classes for managing OpenFOAM cases, from simple synchronous operations to advanced asynchronous and HPC workflows.

Core Classes
------------

FoamCaseBase
~~~~~~~~~~~~

Base class providing common functionality for all case types.

.. autoclass:: foamlib.FoamCaseBase
   :members:
   :show-inheritance:

FoamCase
~~~~~~~~

Main class for synchronous case operations.

.. autoclass:: foamlib.FoamCase
   :members:
   :inherited-members:
   :show-inheritance:

Asynchronous Classes
--------------------

AsyncFoamCase
~~~~~~~~~~~~~

Asynchronous version of FoamCase for concurrent operations.

.. autoclass:: foamlib.AsyncFoamCase
   :members:
   :inherited-members:
   :show-inheritance:

AsyncSlurmFoamCase
~~~~~~~~~~~~~~~~~~

Specialized class for HPC clusters using Slurm job scheduler.

.. autoclass:: foamlib.AsyncSlurmFoamCase
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Case Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from foamlib import FoamCase
   
   # Clone a tutorial case
   case = FoamCase.clone_tutorial(
       "incompressible/simpleFoam/pitzDaily",
       destination="my_case"
   )
   
   # Modify settings
   case.control_dict["endTime"] = 1000
   case.save()
   
   # Run the case
   case.run()
   
   # Access results
   velocity = case[case.latest_time]["U"]

Asynchronous Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from foamlib import AsyncFoamCase
   
   async def run_multiple_cases():
       cases = []
       
       # Create multiple cases
       for i in range(3):
           case = AsyncFoamCase.clone_tutorial(
               "incompressible/simpleFoam/pitzDaily",
               destination=f"case_{i}"
           )
           cases.append(case)
       
       # Run all cases concurrently
       await asyncio.gather(*[case.run() for case in cases])
       
       # Process results
       for case in cases:
           print(f"Case {case.path.name}: Latest time = {case.latest_time}")
   
   # Execute
   asyncio.run(run_multiple_cases())

HPC Integration
~~~~~~~~~~~~~~~

.. code-block:: python

   from foamlib import AsyncSlurmFoamCase
   
   async def submit_hpc_job():
       case = AsyncSlurmFoamCase.clone_tutorial(
           "incompressible/simpleFoam/pitzDaily",
           destination="hpc_case"
       )
       
       # Configure Slurm options
       case.slurm_options = {
           "partition": "compute",
           "nodes": 1,
           "ntasks-per-node": 16,
           "time": "02:00:00",
           "job-name": "foamlib-test"
       }
       
       # Submit and wait for completion
       await case.run()
       print(f"Job completed. Latest time: {case.latest_time}")

Case Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive case setup
   case = FoamCase("my_case")
   
   # Configure time controls
   case.control_dict.update({
       "startTime": 0,
       "endTime": 1000,
       "deltaT": 0.01,
       "writeControl": "timeStep",
       "writeInterval": 100
   })
   
   # Configure solver settings
   case["system/fvSolution"]["SIMPLE"]["nNonOrthogonalCorrectors"] = 2
   case["system/fvSolution"]["relaxationFactors"]["p"] = 0.3
   
   # Set transport properties
   case["constant/transportProperties"]["nu"] = [0, 2, -1, 0, 0, 0, 0, 1.5e-05]
   
   # Save all changes
   case.save()

Result Access
~~~~~~~~~~~~~

.. code-block:: python

   # Access time directories
   print(f"Available times: {case.times}")
   print(f"Latest time: {case.latest_time}")
   
   # Access fields
   latest_U = case[case.latest_time]["U"]
   latest_p = case[case.latest_time]["p"]
   
   # Access specific time
   mid_time = case[500]["U"]  # Velocity at t=500
   
   # Access log files
   if case.path.joinpath("log.simpleFoam").exists():
       print("Solver log available")
   
   # Check convergence
   residuals = case.residuals  # If available
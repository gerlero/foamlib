File Manipulation
=================

foamlib provides comprehensive support for reading, writing, and manipulating OpenFOAM files.

Core Classes
------------

FoamFile
~~~~~~~~

The base class for OpenFOAM dictionary files.

.. autoclass:: foamlib.FoamFile
   :members:
   :inherited-members:
   :show-inheritance:

FoamFieldFile  
~~~~~~~~~~~~~

Specialized class for OpenFOAM field files with support for internal fields, boundary conditions, and dimensions.

.. autoclass:: foamlib.FoamFieldFile
   :members:
   :inherited-members:
   :show-inheritance:

Dimension Handling
------------------

Dimensioned
~~~~~~~~~~~

Class for handling dimensioned scalars, vectors, and tensors.

.. autoclass:: foamlib.Dimensioned
   :members:
   :show-inheritance:

DimensionSet
~~~~~~~~~~~~

Class for representing OpenFOAM dimension sets.

.. autoclass:: foamlib.DimensionSet  
   :members:
   :show-inheritance:

Examples
--------

Basic File Reading
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from foamlib import FoamFile, FoamFieldFile
   
   # Read a dictionary file
   control_dict = FoamFile("system/controlDict")
   print(f"End time: {control_dict['endTime']}")
   
   # Read a field file
   velocity = FoamFieldFile("0/U")
   print(f"Internal field shape: {velocity.internal_field.shape}")

Creating Files
~~~~~~~~~~~~~~

.. code-block:: python

   from foamlib import FoamFile, Dimensioned, DimensionSet
   
   # Create transport properties
   props = FoamFile()
   props["version"] = "2.0"
   props["format"] = "ascii"
   props["class"] = "dictionary"
   props["object"] = "transportProperties"
   
   # Add dimensioned viscosity
   nu = Dimensioned("nu", DimensionSet(0, 2, -1, 0, 0, 0, 0), 1e-6)
   props["nu"] = nu
   
   # Save file
   props.save("constant/transportProperties")

Field Manipulation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from foamlib import FoamFieldFile
   
   # Load velocity field
   U = FoamFieldFile("0/U")
   
   # Modify internal field
   U.internal_field[:, 0] *= 1.1  # Increase x-velocity by 10%
   
   # Save modified field
   U.save("0/U_modified")
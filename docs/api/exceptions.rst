Exceptions
==========

foamlib defines custom exceptions for better error handling and debugging.

CalledProcessError
------------------

.. autoclass:: foamlib.CalledProcessError
   :members:
   :show-inheritance:

This exception is raised when an OpenFOAM application fails during execution.

Example usage:

.. code-block:: python

   from foamlib import FoamCase, CalledProcessError
   
   case = FoamCase("my_case")
   
   try:
       case.run(["blockMesh"])
   except CalledProcessError as e:
       print(f"blockMesh failed with return code {e.returncode}")
       print(f"Command: {' '.join(e.cmd)}")
       if e.stdout:
           print(f"Output: {e.stdout}")
       if e.stderr:
           print(f"Error: {e.stderr}")
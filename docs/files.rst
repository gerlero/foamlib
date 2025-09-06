📄 File manipulation
====================

.. autoclass:: foamlib.FoamFile
   :show-inheritance:

.. autoclass:: foamlib.FoamFile.SubDict
   :show-inheritance:


Field files
-----------

.. autoclass:: foamlib.FoamFieldFile
   :show-inheritance:
   
   .. autoproperty:: foamlib.FoamFieldFile.internal_field
   .. autoproperty:: foamlib.FoamFieldFile.boundary_field
   .. autoproperty:: foamlib.FoamFieldFile.dimensions


.. autoclass:: foamlib.FoamFieldFile.BoundariesSubDict
   :show-inheritance:

.. autoclass:: foamlib.FoamFieldFile.BoundarySubDict
   :show-inheritance:

   .. autoproperty:: foamlib.FoamFieldFile.BoundarySubDict.value
   .. autoproperty:: foamlib.FoamFieldFile.BoundarySubDict.type


Supporting types
----------------

.. autoclass:: foamlib.Dimensioned

.. autoclass:: foamlib.DimensionSet


Standalone parsing/serialization
--------------------------------

.. automethod:: foamlib.FoamFile.loads

.. automethod:: foamlib.FoamFile.dumps
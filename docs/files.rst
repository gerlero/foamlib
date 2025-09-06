📄 File manipulation
====================

.. autoclass:: foamlib.FoamFile
   :show-inheritance:

   .. automethod:: foamlib.FoamFile.__getitem__
   .. automethod:: foamlib.FoamFile.__setitem__
   .. automethod:: foamlib.FoamFile.__contains__
   .. automethod:: foamlib.FoamFile.get
   .. automethod:: foamlib.FoamFile.getone
   .. automethod:: foamlib.FoamFile.getall
   .. automethod:: foamlib.FoamFile.add
   .. automethod:: foamlib.FoamFile.setdefault
   .. automethod:: foamlib.FoamFile.pop
   .. automethod:: foamlib.FoamFile.popone
   .. automethod:: foamlib.FoamFile.popall
   .. automethod:: foamlib.FoamFile.__len__
   .. automethod:: foamlib.FoamFile.__iter__
   .. automethod:: foamlib.FoamFile.keys
   .. automethod:: foamlib.FoamFile.values
   .. automethod:: foamlib.FoamFile.items
   .. automethod:: foamlib.FoamFile.clear
   .. automethod:: foamlib.FoamFile.update
   .. automethod:: foamlib.FoamFile.extend
   .. automethod:: foamlib.FoamFile.merge
   .. automethod:: foamlib.FoamFile.as_dict
   .. automethod:: foamlib.FoamFile.__enter__
   .. automethod:: foamlib.FoamFile.__exit__
   .. automethod:: foamlib.FoamFile.loads
   .. automethod:: foamlib.FoamFile.dumps

   .. autoclass:: foamlib.FoamFile.SubDict
      :show-inheritance:

      .. automethod:: foamlib.FoamFile.SubDict.__getitem__
      .. automethod:: foamlib.FoamFile.SubDict.__setitem__
      .. automethod:: foamlib.FoamFile.SubDict.__contains__
      .. automethod:: foamlib.FoamFile.SubDict.get
      .. automethod:: foamlib.FoamFile.SubDict.getone
      .. automethod:: foamlib.FoamFile.SubDict.getall
      .. automethod:: foamlib.FoamFile.SubDict.add
      .. automethod:: foamlib.FoamFile.SubDict.setdefault
      .. automethod:: foamlib.FoamFile.SubDict.pop
      .. automethod:: foamlib.FoamFile.SubDict.popone
      .. automethod:: foamlib.FoamFile.SubDict.popall
      .. automethod:: foamlib.FoamFile.SubDict.__len__
      .. automethod:: foamlib.FoamFile.SubDict.__iter__
      .. automethod:: foamlib.FoamFile.SubDict.keys
      .. automethod:: foamlib.FoamFile.SubDict.values
      .. automethod:: foamlib.FoamFile.SubDict.items
      .. automethod:: foamlib.FoamFile.SubDict.clear
      .. automethod:: foamlib.FoamFile.SubDict.update
      .. automethod:: foamlib.FoamFile.SubDict.extend
      .. automethod:: foamlib.FoamFile.SubDict.merge
      .. automethod:: foamlib.FoamFile.SubDict.as_dict

   .. class:: foamlib.FoamFile.Dimensioned

      Alias of :class:`Dimensioned` provided for backward compatibility.

      Prefer using :class:`foamlib.Dimensioned` directly.

   .. class:: foamlib.FoamFile.DimensionSet

      Alias of :class:`DimensionSet` provided for backward compatibility.

      Prefer using :class:`foamlib.DimensionSet` directly.

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


Additional types
----------------

.. autoclass:: foamlib.Dimensioned

.. autoclass:: foamlib.DimensionSet

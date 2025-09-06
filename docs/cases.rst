📁 Case manipulation
====================

.. autoclass:: foamlib.FoamCaseBase
   :show-inheritance:

   .. autoproperty:: foamlib.FoamCaseBase.name
   .. autoproperty:: foamlib.FoamCaseBase.control_dict
   .. autoproperty:: foamlib.FoamCaseBase.block_mesh_dict
   .. autoproperty:: foamlib.FoamCaseBase.decompose_par_dict
   .. autoproperty:: foamlib.FoamCaseBase.fv_schemes
   .. autoproperty:: foamlib.FoamCaseBase.fv_solution
   .. autoproperty:: foamlib.FoamCaseBase.transport_properties
   .. autoproperty:: foamlib.FoamCaseBase.turbulence_properties
   .. autoproperty:: foamlib.FoamCaseBase.application
   .. automethod:: foamlib.FoamCaseBase.file
   .. automethod:: foamlib.FoamCaseBase.__getitem__
   .. automethod:: foamlib.FoamCaseBase.__iter__
   .. automethod:: foamlib.FoamCaseBase.__contains__
   .. automethod:: foamlib.FoamCaseBase.__len__

   .. autoclass:: foamlib.FoamCaseBase.TimeDirectory
      :show-inheritance:

      .. autoproperty:: foamlib.FoamCaseBase.TimeDirectory.name
      .. autoproperty:: foamlib.FoamCaseBase.TimeDirectory.time
      .. automethod:: foamlib.FoamCaseBase.TimeDirectory.__getitem__
      .. automethod:: foamlib.FoamCaseBase.TimeDirectory.__iter__
      .. automethod:: foamlib.FoamCaseBase.TimeDirectory.__contains__
      .. automethod:: foamlib.FoamCaseBase.TimeDirectory.__len__


.. autoclass:: foamlib.FoamCase
   :members:
   :show-inheritance:

.. autoclass:: foamlib.AsyncFoamCase
   :members:
   :show-inheritance:

.. autoclass:: foamlib.AsyncSlurmFoamCase
   :members:
   :show-inheritance:


Exceptions
----------

.. autoclass:: foamlib.CalledProcessError
   :members:

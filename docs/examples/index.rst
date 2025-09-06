Examples Gallery
================

This section contains detailed examples showing how to use foamlib for various OpenFOAM workflows.

Basic Examples
--------------

.. toctree::
   :maxdepth: 1

   basic_usage
   file_manipulation  
   case_management

Advanced Examples
-----------------

.. toctree::
   :maxdepth: 1

   parametric_studies
   async_operations
   hpc_integration
   postprocessing_examples

Real-World Applications
-----------------------

.. toctree::
   :maxdepth: 1

   optimization
   validation_studies
   batch_processing

Complete Workflows
------------------

The following examples demonstrate complete workflows from case setup to result analysis:

* :doc:`../example` - Basic diffusion validation case
* :doc:`../parametricstudy` - Setting up parametric studies  
* :doc:`../postprocessing` - Advanced post-processing techniques

Code Repository
---------------

All example code is available in the `examples directory <https://github.com/gerlero/foamlib/tree/main/examples>`_ of the foamlib repository.

.. note::
   To run the examples, you'll need to install the examples dependencies:
   
   .. code-block:: bash
   
      pip install foamlib[examples]
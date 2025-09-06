.. image:: https://github.com/gerlero/foamlib/blob/main/logo.png?raw=true
   :alt: foamlib logo
   :width: 200 px
   :target: https://github.com/gerlero/foamlib
   :align: center

=========================================================================

**foamlib** is a modern Python package that simplifies working with OpenFOAM cases and files by providing a standalone parser for seamless interaction with OpenFOAM's input/output data. It includes robust case-handling capabilities that reduce boilerplate code and enable efficient pre-processing, post-processing, and simulation management directly from Python.

With support for ASCII- and binary-formatted fields, a fully type-hinted API, and asynchronous operations, foamlib offers a streamlined, Pythonic approach to automating and managing OpenFOAM workflows.

.. note::
   foamlib is published in the Journal of Open Source Software (JOSS). 
   `Read the paper <https://doi.org/10.21105/joss.07633>`_ for more details about the design and capabilities.

Key Features
============

🚀 **Easy to Use**
   Simple, Pythonic interface for OpenFOAM case management

⚡ **High Performance** 
   Fast parsing with support for both ASCII and binary formats

🔧 **Comprehensive**
   File manipulation, case running, and result processing in one package

🔄 **Asynchronous Support**
   Concurrent case execution with async/await syntax

🏗️ **HPC Ready**
   Built-in support for Slurm-based HPC clusters

🔗 **Well Integrated**
   Works seamlessly with NumPy, pandas, and the scientific Python ecosystem

Quick Start
===========

Installation
------------

Install foamlib from PyPI:

.. code-block:: bash

   pip install foamlib

Or with conda:

.. code-block:: bash

   conda install -c conda-forge foamlib

Basic Usage
-----------

.. code-block:: python

   import foamlib

   # Clone a tutorial case
   case = foamlib.FoamCase.clone_tutorial("incompressible/simpleFoam/pitzDaily")
   
   # Run the case
   case.run()
   
   # Access results
   velocity = case[200]["U"]  # Velocity field at time 200
   print(f"Max velocity: {velocity.max()}")

📚 Documentation Guide
=======================

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   
User Guide
----------

.. toctree::
   :maxdepth: 1
   
   example
   parametricstudy
   postprocessing

API Reference  
-------------

.. toctree::
   :maxdepth: 1

   files
   cases
   api/exceptions

Advanced Topics
---------------

.. toctree::
   :maxdepth: 1
   
   examples/index
   tutorials/index
   
Project Information
-------------------

.. toctree::
   :maxdepth: 1
   
   changelog

🔍 Indices and Search
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

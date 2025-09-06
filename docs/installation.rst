Installation
============

foamlib can be installed in several ways, depending on your needs and preferences.

Requirements
------------

* Python 3.8 or later
* OpenFOAM (any version) - required only for running cases, not for file manipulation
* Operating System: Linux or macOS (Windows support via WSL)

Basic Installation
------------------

PyPI (Recommended)
~~~~~~~~~~~~~~~~~~

Install the latest stable release from PyPI:

.. code-block:: bash

   pip install foamlib

This installs the core foamlib package with basic functionality for file manipulation and case management.

Conda/Mamba
~~~~~~~~~~~

Install from conda-forge:

.. code-block:: bash

   conda install -c conda-forge foamlib

Or with mamba:

.. code-block:: bash

   mamba install -c conda-forge foamlib

Optional Dependencies
---------------------

foamlib offers several optional dependency groups for extended functionality:

Preprocessing
~~~~~~~~~~~~~

For parametric studies and advanced case setup:

.. code-block:: bash

   pip install foamlib[preprocessing]

This includes:

* ``pandas`` - for data manipulation in parametric studies
* ``pydantic`` - for configuration validation

Postprocessing  
~~~~~~~~~~~~~~

For advanced result analysis and visualization:

.. code-block:: bash

   pip install foamlib[postprocessing]

This includes:

* ``pandas`` - for data analysis
* ``defusedxml`` - for safe XML parsing

Examples
~~~~~~~~

To run all the examples and tutorials:

.. code-block:: bash

   pip install foamlib[examples]

This includes preprocessing, postprocessing, and additional packages like ``scipy``.

All Features
~~~~~~~~~~~~

To install everything:

.. code-block:: bash

   pip install foamlib[preprocessing,postprocessing,examples]

Development Installation
------------------------

For developers who want to contribute to foamlib:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/gerlero/foamlib.git
      cd foamlib

2. Install in development mode:

   .. code-block:: bash

      pip install -e .[dev]

This installs all dependencies needed for development, including testing, linting, and documentation tools.

Docker Installation
-------------------

A Docker image with foamlib and OpenFOAM is available:

.. code-block:: bash

   docker pull microfluidica/foamlib

This provides a complete environment with both foamlib and OpenFOAM pre-installed.

Verification
------------

To verify your installation, run:

.. code-block:: python

   import foamlib
   print(foamlib.__version__)

You can also run a quick test:

.. code-block:: python

   import foamlib
   
   # Test file manipulation (doesn't require OpenFOAM)
   from foamlib import FoamFile
   
   # Create a simple dictionary
   foam_dict = FoamFile()
   foam_dict["version"] = "2.0"
   foam_dict["format"] = "ascii"
   print("✓ foamlib installation successful!")

Troubleshooting
---------------

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure you have the required dependencies:

.. code-block:: bash

   pip install --upgrade foamlib

Missing OpenFOAM
~~~~~~~~~~~~~~~~~

foamlib can manipulate OpenFOAM files without OpenFOAM installed, but you need OpenFOAM to run cases. If you get errors when trying to run cases:

1. Ensure OpenFOAM is properly installed and sourced
2. Check that OpenFOAM commands are available in your PATH:

   .. code-block:: bash

      which simpleFoam

Version Conflicts
~~~~~~~~~~~~~~~~~

If you have conflicting package versions, consider using a virtual environment:

.. code-block:: bash

   python -m venv foamlib-env
   source foamlib-env/bin/activate
   pip install foamlib

Getting Help
------------

If you encounter issues:

1. Check the `GitHub issues <https://github.com/gerlero/foamlib/issues>`_
2. Start a `discussion <https://github.com/gerlero/foamlib/discussions>`_
3. Consult the :doc:`quickstart` guide for basic usage examples
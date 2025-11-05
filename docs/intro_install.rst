===========================================
Introduction to "Bacteria Density Analyzer"
===========================================

1. Introduction
===============

- :code:`Bacteria Density Analyzer` is a Python module and a Napari plugin allowing to segment, skeletonize and extract a 1D signal from a 3D filament.
- It uses a segmentation channel (nuclei) and N fluorescence channels (bacteria) to compute the density of bacteria along the filaments.
- This module offers the possibility to have several filaments on the same image, even if they happen to be broken or if a region can't be segmented.
- The output is a CSV file containing, for each filament, the density of bacteria along its length for each fluorescence channel.
- The plots corresponding to measures are also exported.
- Measures are made along a 1D skeleton and are binned according to a user-defined length.

.. image:: _images/overview.png
  :align: center

2. Install the plugin 
=====================

- We strongly recommend to use `conda <https://docs.conda.io/en/latest/miniconda.html>`_ or any other virtual environment manager instead of installing Napari and :code:`bacteria-density` in your system's Python.
- Napari is only required if you want to use bacteria-density with a graphical interface.
- Napari is not part of bacteria-density's dependencies, so you will have to install it separately.
- Each of the commands below is supposed to be run after you activated your virtual environment.
- If the install was successful, you should see the plugin in Napari in the top bar menu: Plugins > Density vs. distance.

A. Development versions
-----------------------

If you want to install a version from GitHub, you need to have `Git <https://git-scm.com/install/>`_ installed on your computer.

+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Method                | Instructions                                                                                                                                                                             |
+=======================+==========================================================================================================================================================================================+
| Latest                | :code:`pip install git+https://github.com/MontpellierRessourcesImagerie/bacteria-density.git`                                                                                            |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| From an archive       | - Download `the archive <https://github.com/MontpellierRessourcesImagerie/bacteria-density/archive/refs/heads/master.zip>`_  :code:`pyproject.toml` and launch :code:`pip install -e .`. |
|                       | - From the terminal containing your virtual env, move to the folder containing the file :code:`pyproject.toml`                                                                           |
|                       | - Run the command :code:`pip install -e .`                                                                                                                                               |
+-----------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

B. Stable versions
------------------

+-----------------------+------------------------------------------------------------------------------------+
| Method                | Instructions                                                                       |
+=======================+====================================================================================+
| pip                   | Activate your conda environment, and type :code:`pip install bacteria-density`.    |
+-----------------------+------------------------------------------------------------------------------------+

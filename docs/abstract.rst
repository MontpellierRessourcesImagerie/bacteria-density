===================================
Bacteria Density Analyzer: Abstract
===================================

0. User provided data
=====================

A. Images
---------

- At first, the user is expected to provide several channels belonging to a 3D image.
- The mandatory channel is the one we will use to segment the filaments. During the tests and the development of this module, we used a nuclei staining (DAPI).
- The other channels are fluorescence channels corresponding to bacteria stains (for example GFP, RFP, etc.).
- The module can handle an arbitrary number of fluorescence channels at the same time.
- The image can contain several filaments, even if they are broken or partially segmented.

B. Regions of interest
----------------------

- As images contain several filaments, the user must indicate what part of the image belongs to which filament.
- To do so, the user must draw polygons around each filament using Napari's :code:`Shapes` layer.
- The color associated to each polygon will define the filament's ID.
- A filament can be broken into several parts, each part being surrounded by a polygon of the same color.
- On the example image below, there are two filaments/branches (the red and the green ones), the first being broken into three parts and the second being monolithic.
- On the example, the three regions of the red filament are adjacent, but in reality, breaking a filament should only be done if a part of it can't be segmented (for example because of low signal-to-noise ratio), so the different parts can be far from each other.
- The plugin will by itself re-link the different chunks in the final output.

.. image:: _images/broken.png
  :align: center

C. Starting points
------------------

- As a filament can be broken in any number of parts, the user must provide for each branch (== each color) one single point that approximately corresponds to the start of the filament.
- From there, the plugin will rebuild the full filament by connecting all the different parts together.
- The starting points have to be placed inside the corresponding polygons.
- The exact location doesn't matter as it is just a proximity indicator. The path of every filament will be processed and then, we will search for the closest point to the starting point provided by the user.


1. Workflow description
=======================

A. Build a mask of filaments
----------------------------

- Everything starts by segmenting the filaments in 3D using the segmentation channel provided by the user.
- A binary mask is created and fixed, allowing us to know which pixels are part of the filaments and which are not.
- If a pixel is positive in the mask, its intensity will be used in the computation of the integrated intensity and density.

.. image:: _images/make_mask.png
  :align: center

B. Skeletonize filaments
------------------------

- From the 3D binary mask, we compute a skeleton of the filaments.
- It allows to reduce the 3D structure of the filaments to a 1D path going through the center of the filaments.
- A skeleton is typically represented as a binary mask containing 1-pixel wide lines going through the center of the filaments.
- We will base ourselves on this skeleton to compute the length of the filaments and to extract the intensity values along them.

.. image:: _images/skeleton.png
  :align: center

C. Extract medial path
----------------------

- Depending on the mask and its level of details, the skeleton can contain small branches and loops that are not part of the real filaments.
- To avoid these artifacts, we extract a medial path going through the skeleton from one end to the other.
- To extract this path, we start by extracting the pixel corresponding to the starting point for each chunk.
- From this pixel that we consider as the root of the filament, we search for the longest path going through the skeleton.
- This way, we avoid small branches and loops that could be present in the skeleton.

.. image:: _images/medial.png
  :align: center

D. Measures
-----------

- We now have a 3D mask indicating which pixels belong to filaments, and a 1D path going through the center of each filament.
- The goal is to project (in some way) the 3D information onto a 1D object.
- To do so, we take each 3D coordinate from the mask and search the closest point on the 1D path.
- For each point of the 1D path, we create an accumulator that will store the integrated intensity (== the sum of all the intensities of the voxels that project onto this point).
- Using the same principle, we can extract the volume of each accumulator by counting the number of voxels that project onto each point of the 1D path.
- Finally, the density is computed by dividing the integrated intensity by the volume for each point of the 1D path.
- Exporting such values would result in rather noisy curves, so we bin the values according to a user-defined length (for example 1 micron).
- It basically means that we group the 1D points by chunks of 1 micron and gather the statistics by chunk.
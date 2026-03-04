=======================================
Bacteria Density Analyzer: User's guide
=======================================

0. Load the images and open the plugin's panel
==============================================

- To use an image, the channels must be separated in different layers of Napari.
- The easier way to get that is to have your images split into different files, each file containing one channel.
- Another way would be to use :code:`napari-imagej` (another plugin) that allows Napari to read the memory of Fiji.
- If you already have one TIFF file per channel, you can drag'n'drop then into Napari's main window.
- They should appear in the "layers list" (in the lower left corner). The list is a stack, so you only see the uppermost layer.
- You can use the "once" button in the "layer controls" panel to adjust the contrast of each layer for better visualization.
- You can now go in the top bar menu: Plugins > Density vs. distance to open the plugin's panel.
- The panel should appear on the right side of Napari's window.

1. Segmentation
===============

- In the "Nuclei channel" dropdown menu, select the layer corresponding to the nuclei channel.

2. Measures
===========

- For each channel in which you want to make intensity measurements:
    - Use the "Ci" dropdown menu to select the corresponding layer.
    - In the text input on the right containing "---" by default, provide the name for this channel that will be used in the results table.
- There is always a blank line at the end, it is normal.

3. ROIs
=======

- Above the "layers list" panel, add a new "Shape layer".
- Add polygons for each region that you would like to process. Before drawing, you can increase the edge thickness for better visibility.
- Use the `F1`, `F2`, `F3` & `F4` buttons to change the edge color of these polygons. Each color represents a filament. If several areas share a same color, it means that they are part of the same filament and were cut for some reason (obstacle, crossing, ...).
    - 1: Red
    - 2: Green
    - 3: Blue
    - 4: Yellow
- You must click on one of these buttons after drawing **each** polygon, it is not possible to change the color of a polygon that is not the last one.
- Add a new "Points layer". In the "layers control" panel, switch to the "Add points" mode. Before placing the first point, you can increase the size of the points for better visibility.
- For each filament (each color), add a point approximately where the filament starts. There should be exactly one point per color.
- In the dropdown menus of the ROIs box of the plugin, provide the new shape and point layers.

**Notes about drawing polygons:**

- If some areas are folded on themselves, the polygon must pass where the separation should be.
- If some debris are stuck to the object, the polygons should "shave" them off.
- If you have darker areas longer than twice the kernel size, they should be in their own polygons.
- Finish the polygons by a rounded or pointy end rather than a flat one to guide the medial path and make it create an angle.
- A polygon should slightly overlap with the next to improve the continuity (if possible).
- The segments of a polygon should not intersect with each other (different polygons can overlap, that's not a problem).
- To connect the medial paths together, we look for the closest point in another polygon from the end of the filament in the current polygon. If you don't pay attention to that, the linking could be made with the wrong polygon or the wrong end of the filament.

4. Settings
===========

- Use the "Set output folder" button to provide the path to an empty directory.
- Note: If you already completed an analysis and need to reload the results (for visualization only), you can use this same button and provide the path to the folder that you used for the desired analysis.
- Select the binning length using the input below (binning of measures along the medial path).
- Select the list of measurements that you would like to process.
- A logarithm function is applied to intensities to reduce the dynamic range. The higher the "Log factor", the more the dynamic range is reduced. You can adjust it to get a better segmentation. If you have areas with a lot of contrast, you should increase it.
- The threshold applied to the image is the mean of intensities in the polygon. To have a finer control on the segmentation, you can multiply this threshold by a factor. If you want to have a more inclusive segmentation, you should decrease it.
- In an attempt to fill holes, a morphological closing is applied to the mask: the "Kernel size" is the size of the structuring element used for this closing. The bigger the holes, the bigger the kernel size should be. If you increase it too much, you will merge folded areas and details.

5. Workflow
===========

- From this point, you can click on each button from top to bottom:
- **Chunk images:** Will export a copy of each selected area in the working directory.
- **Skeletonize:** Will create a mask and a skeleton of the corresponding branch. An outline of the mask will be displayed.
- **Make medial path:** Will process the path over the organ and assemble the fragments. You can check the result by searching for the middle of the stack using the slider under the image.
- **Measure:** Will perform the measures and export the CSV.
- **Create Plots:** Export the plots as PNG with a dashed-line between segments.

+-----------------------+------------------------------------------------------------------------------------+
| Metric                | Description                                                                        |
+=======================+====================================================================================+
| Integrated intensity  | Sum of all intensities for the voxels refering to this bin.                        |
+-----------------------+------------------------------------------------------------------------------------+
| Integrated volume     | Total volume taken by the voxels refering to this bin.                             |
+-----------------------+------------------------------------------------------------------------------------+
| Local width           | Local width measured along the medial path.                                        |
+-----------------------+------------------------------------------------------------------------------------+
| Density               | Local density along the medial path: integrated intensity normalized by volume.    |
+-----------------------+------------------------------------------------------------------------------------+

6. Optional
===========

- If you need so, you can get an approximation of the discarded volume of organ on your image.
- To do that, add a new shape layer and draw a polygon over each area that you didn't include in any filament.
- In the "Discarded polygons" dropdown menu from the plugin's tab, choose the shape layer containing the polygons that you just drew.
- If you click the "Process discarded volume" button, the discarded volume should be displayed right beneath it.
- It is an approximation in the way that the area you just indicated cannot be segmented. To estimate the volume, an artificial mask is built. It is as deep (== Z-axis == nb of slices) as the whole stack and has the polygons shape on every slice.

# Napari bacteria density

<img width="3800" height="2122" alt="nbd" src="https://github.com/user-attachments/assets/e748fbff-72c4-4c90-8026-5f1558e4d743" />

## Install

- Make sure that you installed [Git](https://git-scm.com/downloads) on your system.
- Make sure that you installed a Python environments manager (like [Miniconda](https://repo.anaconda.com/miniconda/)).
- Open a terminal and create a new environment using the command `conda create -n bacteria-density -y python=3.10`.
- Activate the newly created environment using `conda activate bacteria-density`.
- Install the development version of the plugin using `pip install git+https://github.com/MontpellierRessourcesImagerie/bacteria-density.git`.
- If you want to use it through a GUI, install Napari alongside it with `pip install napari[all]`.
- To calibrate your images, you will need to install the calibration tool using `pip install set-calibration`.
- If your images are not TIFF, you will need to install ImageJ's bridge using `pip install napari-imagej`.

## Usage

- Open a new terminal and activate the environment containing Napari using `conda activate bacteria-density`.
- Launch Napari with the command `napari`.
- In the top-bar, within the "Plugins" menu, you should find `ImageJ2`, `Scale tool` & `Density vs. distance`.
- Start by using `ImageJ2` to open your image and import it in Napari. In the left column, you should now see one layer per channel. You can rename them as you wish.
- You can now close the `ImageJ2` panel.
- You should now open `Scale tool` to provide the physical size of voxels. Don't forget that in Napari, the order is ZYX instead of XYZ.
- Once you're done providing the scale, you can close the `Scale tool` panel.
- You can now open the `Density vs. distance` panel.

## Process the image

#### 1. Segmentation

- In the dropdown menu, select the layer corresponding to the nuclei.

#### 2. Measures

- For each channel in which you want to make intensity measurements, indicate the layer (in the dropdown menu) and provide a name for what it contains.

#### 3. ROIs

- Above the layers list, add a new "Shape layer".
- Add rectangles for each region that you would like to process.
- Change the "edge color" to represent the different filaments (one color == one filament).
- Add a new "Points layer".
- For each filament (one box per color), add a point approximately where the filament starts.
- In the dropdown menus, provide the new shape and point layers.

#### 4. Settings

- Using the "Set output folder", provide the path to an empty directory.
- Select the binning length using the input below (binning of measures along the skeleton).
- Select the list of measurements that you would like to process.

#### 5. Workflow

- **Chunk images:** Will export a copy of each selected area in the working directory.
- **Skeletonize:** Will create a mask and a skeleton of the corresponding branch. An outline of the mask will be displayed.
- **Make medial path:** Will process the path over the organ and assemble the fragments.
- **Measure:** Will perform the measures and export the CSV.
- **Create Plots:** Export the plots as PNG with a dashed-line between segments.


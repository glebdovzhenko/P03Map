# P03Map
This software does single peak refinement in sequential diffraction experiments.

Developed by Gleb Dovzhenko for Helmholtz-Zentrum Hereon.

Licensed under GNU GPL v.3.

## Installation

### Windows:
Download the `.zip` archive from the latest release [here](https://github.com/glebdovzhenko/P03Map/releases). 
Extract the archive and launch `P03Map.bat`.

### Other systems:
Only installation from source is available. `requirements.yml` contains all the necessary packages and can be used by conda to create a virtual environment.

## Usage

### Data format
P03Map accepts `FIO` files produced by the P03 beamline and data files produced by pyFAI from the detector images.
The software collects the line scans (each `FIO` + subfolder with `.dat` represent one scan) into a 2D map in accordance with the coordinate data in the `FIO` files.

### Import
Use `File -> Import` menu to import the FIO files. 
Select all the `FIO` files from the map you want analysed.
Importing the next batch of files will clear the current data.

As this process may take some time, the status bar at the bottom of the window shows progress.
Once import is complete, the plot on the `Mean spectra` tab will show the diffraction pattern averaged over all imported data.

### Peak selection
Once the import is done, you can select a peak you want to fit using the area selection. 
The red line inside the selection shows a Gaussian + linear estimate of the selected data using the algorithm by [Roonizi (2013)](https://doi.org/10.1109/LSP.2013.2280577).

Move the selection until the peak you want to observe has a relatively good estimate. 
This estimate will be used as a starting point for refinement.

### Peak refinement
Go to the `Map` tab.

If you press `Refine`, the refinement of all imported data will start. 
Each spectrum is refined with a Gaussian (peak) + linear (background) functions.

The progress is shown on the progress bar below.
You can use the `Stop` button to stop the refinement if it is taking too long and go back to the `Mean spectra` tab to improve the estimate or select another peak.
The text below the progress bar shows how many refinements were successful. 
Unsuccessful refinements will be omitted from the plot.

### Data presentation
Once the refinement is complete, you will see the results on the 3D plot.
You can make changes to the plot using the combo boxes:
1. `Data`: choose peak `Height`, `Width`, or `Center` to be plotted.
2. `Scale`: choose `z` axis scale: `Linear`, `Log`, or `Root`.
3. `Plot`: choose between a coloured 3D plot (`Landscape`) or a heat map (`Flat`).
4. `Interpolation`: represent data as actual datapoints (`Points`), or choose an interpolation (`Nearest` / `Linear` / `Cubic`).
5. `Colormap`: choose the colormap.

To manipulate the 3D plot use the mouse wheel to zoom in / out and left button drag to rotate the scene. 
`Ctrl` /  `⌘` + left button drag translates the scene.

You can use the `Side view` / `Top view` buttons to quickly jump between convenient viewpoints. 
`Top view` is intended to be used with `Plot`: `Flat` to produce heat maps.

### Comparing peaks
Once you have completed the refinement on one peak, you can go back to the `Mean spectra` tab, select another peak, and refine it the same way.
Note that once you press `Refine`, current refinement data is lost.
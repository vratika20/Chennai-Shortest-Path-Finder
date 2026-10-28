Chennai Shortest Path Finder

A GPU-accelerated shortest-path visualization project using Python, OpenCV, and CuPy to explore a Chennai road map.


 Overview

This project processes a Chennai map image, detects potential road pixels using computer vision, and attempts to calculate a shortest path between St. Thomas Mount and Marina Beach using GPU computing.

Technologies Used

- Python
- OpenCV
- NumPy
- CuPy
- CUDA
- Matplotlib
- ipywidgets
- Google Colab

 Project Workflow

1. Upload a Chennai road map image.
2. Process the image to create a binary road mask.
3. Calibrate the map coordinates using interactive sliders.
4. Calculate pixel-based path costs using a custom CUDA kernel.
5. Visualize the resulting path.

 Output Screenshots

 Input Map

Add your input map image here.

 Map Calibration

Add a screenshot showing the calibration sliders here.

Shortest Path Result

Add your actual shortest-path output image here.

How to Run

1. Open Google Colab.
2. Select a GPU runtime if available.
3. Install the required Python dependencies.
4. Upload the project script and a Chennai map image.
5. Adjust the calibration sliders.
6. Run the pathfinder and inspect the result.

Limitations

- Road detection depends on the map's colors.
- Map coordinates require calibration.
- The route is calculated over image pixels, not a real street network.
- GPU execution requires a compatible CUDA environment.

Future Improvements

- Improve road segmentation.
- Allow users to select start and destination points.
- Compare CPU and GPU performance.
- Integrate actual road-network data.
Author

Developed as a Python, computer vision, and GPU-computing project.


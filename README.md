# Installation
1. Install Python. This was implemented using Python 3.10 but other versions should also work.
2. Clone the repository
3. (Optional but recommended) Create a virtual environment and activate it
```
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux / Mac
source ./venv/Scripts/activate
```
4. Install the requirements
```
python -m pip install -r requirements.txt
```

# Usage
```
python mosaic.py <TILE_DIRECTORY> <TARGET_IMAGE> <OUTPUT_IMAGE> --tile-size "(COLUMNS, ROWS)" --target-image-size "(COLUMNS, ROWS)"
```

Example:
```
python mosaic.py .\tiles image.png tiled_image.png --tile-size "(50, 50)" --target-image-size "(2000, 2000)"
```

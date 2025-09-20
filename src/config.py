from pathlib import Path

INPUT_PATH = Path('static/uploads')
PREPROCESSED_PATH = Path('static/processed')
PLOT_PATH = Path('static/plots')
OUTPUT_PATH = Path('outputs')

DEVICE = 'cpu'
MODELS_PATH = Path('models')
SEGMENTATION_MODEL = 'coverage_segmentator.pt'
BM_REGRESSOR_MODEL = 'BM_regressor.pt'
TARGET_SIZE = (512, 256) # width, height
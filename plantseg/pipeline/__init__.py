import logging
from plantseg import plantseg_global_path
import sys
import os

gui_logger = logging.getLogger("PlantSeg")
# hardcode the log-level for now
gui_logger.setLevel(logging.INFO)

# Add console handler (should show in GUI and on the console)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
stream_handler.setFormatter(formatter)
gui_logger.addHandler(stream_handler)

# allowed h5 keys
H5_KEYS = ["raw", "predictions", "segmentation"]

# Resources directory
RESOURCES_DIR = "resources"
raw2seg_config_template = os.path.join(plantseg_global_path, RESOURCES_DIR, "raw2seg_template.yaml")

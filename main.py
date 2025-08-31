# main.py

import os
import random
import tensorflow as tf
import warnings
import gc
import numpy as np # Needed for np.seterr

# Import the high-level components
import config
from pipeline_stages import main_orchestrator

# ==========================================================
# Global Environment and Warning Setup
# ==========================================================
# Suppress TensorFlow informational messages
tf.get_logger().setLevel('ERROR')

# Ignore common, non-critical warnings from data science libraries
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set default NumPy behavior for division-by-zero and invalid values
np.seterr(divide='ignore', invalid='ignore')

# ==========================================================

# ==========================================================
# Set Seeds for Reproducibility
# ==========================================================
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)
random.seed(config.RANDOM_STATE)
# ==========================================================

def setup_environment():
    """Configures TensorFlow strategy based on available hardware."""
    strategy = tf.distribute.MirroredStrategy()
    if config.ENV_PARAM == 1: # Kaggle environment
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            strategy = tf.distribute.TPUStrategy(tpu)
            print("✅ Running on TPU:", tpu.master())
        except ValueError:
            print("✅ Running on GPU/CPU (TPU not detected).")
    else:
        print("✅ Running on local CPU/GPU.")
    return strategy

def main():
    """The main execution function for the entire program."""
    # Set up the distributed strategy
    strategy = setup_environment()

    # Pass the strategy object to the main pipeline
    main_orchestrator(strategy)

    gc.collect()

if __name__ == "__main__":
    # Just call the main function
    main()
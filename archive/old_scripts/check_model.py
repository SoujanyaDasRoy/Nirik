import sys
sys.path.insert(0, 'backend')
from core import inference
import os
print('BASE_DIR:', inference.BASE_DIR)
print('MODEL_PATH:', inference.MODEL_PATH)
print('Exists:', os.path.exists(inference.MODEL_PATH))
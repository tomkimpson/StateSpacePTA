from pathlib import Path
import os 
import math

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
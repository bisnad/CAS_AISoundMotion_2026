import numpy as np
import pickle
import re

class PKL_Tools:

    def load(self, file_name):
        """Loads the raw PKL dictionary from a pickled exported file."""
        with open(file_name, 'rb') as f:
            pkl_data = pickle.load(f)
            
        return pkl_data


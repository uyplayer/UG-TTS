import os.path

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from common.path_config import data_dir




if __name__ == "__main__":

    input_path = os.path.join(data_dir,"LJ001-0001.wav")
    output_path = os.path.join(data_dir,"output.wav")

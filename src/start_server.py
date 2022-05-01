

import gdown
import os
from pathlib import Path


def download_model():
    script_dir = os.path.dirname(__file__)
    rel_path = "app/model/model.pt"
    abs_file_path = os.path.join(script_dir, rel_path)
    path = Path(abs_file_path)

    url = "https://drive.google.com/uc?id=1ExkkBYxpuWHb_ffUzGbArgLCbay77_Ip"

    if path.is_file() == False:
        gdown.download(url=url, output="./app/model/model.pt", quiet=False)
    else:
        print('Model file already downloaded')


download_model()

if __name__ == "__main__":
    os.system('uvicorn main:app --reload')

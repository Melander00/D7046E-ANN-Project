# D7046E ANN Project

## Installation
For some reason `torchaudio` and `torchcodec` doesn't work on newer version. Make sure python 3.10 is installed to allow torch v2.2.2 to be installed.


### Setup environment
Create new virtual environment
```bash
py -3.10 -m venv .venv
```

Activate venv (once every time a new terminal is opened)
```bash
# Linux
source .venv/bin/activate

# Windows
.\.venv\Scripts\activate

# MacOS
?
```

(Optional) Check if venv is correctly setup
```bash
# Linux
which python # Path should include .venv

# Windows
where python # First entry should end with .venv\Scripts\python.exe

# MacOS
?
```

Install dependencies
```bash
pip install -r requirements_3_10.txt
```

### Download datasets
ESC-50 is required to be downloaded. [Download the set here](https://github.com/karoldvl/ESC-50/archive/master.zip), extract the files and copy paste the `audio` folder into root with the name `noise_dataset`.

Python will automatically download the Google SpeechCommand dataset and unzip it into root when running for the first time.
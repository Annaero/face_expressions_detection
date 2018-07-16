# face_expressions_detection
Library and command line util for face expressions detection.

## Instalation

Tested on fresh Ubuntu 18.04 LTS

Install git: sudo apt-get install git
Install anaconda:
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh

Install libraries needed for dlib:
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev

I installed anaconda in my home dir and added aliases:
Далее apip и apython это 
alias apython="~/anaconda3/bin/python"

Download and install face_expressions_detection
git clone https://github.com/Annaero/face_expressions_detection.git
cd face_expressions_detection/
apython setup.py install


## Usage

usage: $ANACONDA_DIR$/bin/expression_detector [-h] [--landmarks] folder

positional arguments:
  folder       folder with images to detect

optional arguments:
  -h, --help   show this help message and exit
  --landmarks  use precalculated landmark files instead of images

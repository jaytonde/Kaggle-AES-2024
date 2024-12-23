#!/bin/bash
<<com
Supports colab, kaggle, paperspace, lambdalabs and jarvislabs environment setup
Usage:
bash setup.sh <ENVIRON> <download_data_or_not>
Example:
bash setup.sh jarvislabs true

!apt-get install dos2unix
com

ENVIRON=$1
DOWNLOAD_DATA=$2
PROJECT="Kaggle-AES-2024"


# get source code from GitHub
git config --global user.name 
git config --global user.email 
git clone 

if [ "$1" == "colab" ]
then
    cd /content/$PROJECT
    
elif [ "$1" == "kaggle" ]
then
    cd /kaggle/working/$PROJECT

elif [ "$1" == "paperspace" ]
then
    cd /notebooks/$PROJECT

else
    echo "Unrecognized environment"
fi

# install deps
pip install -r requirements.txt
source .env
export KAGGLE_USERNAME=$KAGGLE_USERNAME
export KAGGLE_KEY=$KAGGLE_KEY

# change the data id as per the experiment
if [ "$DOWNLOAD_DATA" == "true" ]
then
    mkdir input/
    cd input/
    kaggle datasets download -d jaytonde/aes-dataset
    unzip aes-dataset.zip
    rm aes-dataset.zip
else
    echo "Data download disabled"
fi

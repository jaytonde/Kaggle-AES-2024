#!/bin/bash
<<com
Supports colab, kaggle, paperspace, lambdalabs and jarvislabs environment setup
Usage:
bash setup.sh <ENVIRON> <download_data_or_not>
Example:
bash setup.sh jarvislabs true
com

ENVIRON=$1
DOWNLOAD_DATA=$2
PROJECT="Kaggle-AES-2024"


# get source code from GitHub
git config --global user.name "jaytonde"
git config --global user.email "jaytonde05@gmail.com"
git clone https://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/jaytonde/Kaggle-AES-2024.git

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
pip install -r requirements.txt --upgrade
source .env
export KAGGLE_USERNAME=$KAGGLE_USERNAME
export KAGGLE_KEY=$KAGGLE_KEY

# change the data id as per the experiment
if [ "$DOWNLOAD_DATA" == "true" ]
then
    mkdir input/
    cd input/
    kaggle datasets download -d nbroad/persaude-corpus-2
    unzip persaude-corpus-2.zip
    rm persaude-corpus-2.zip
else
    echo "Data download disabled"
fi
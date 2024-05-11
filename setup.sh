
source .env


echo "Setting up root dir"
#Setup directory based on platform
platform = "paperspace"
if   platform == "paperspace":
     root_dir  = "/notebooks"
elif platform == "kaggle":
     root_dir  = "/kaggle/working"
elif platform == "colab":
     root_dir  = "/content"

cd root_dir
echo "Setting up root dir completed and current wd is $(root_dir)"

echo "Downloading the dataset..."
pip install kaggle
export KAGGLE_USERNAME = KAGGLE_USERNAME
export KAGGLE_KEY      = KAGGLE_KEY

kaggle datasets download -d cdeotte/brain-spectrograms
sudo apt install unzip
unzip zip_path
echo "Dataset download completed..!"

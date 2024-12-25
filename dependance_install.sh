# 1. install NVIDIA Graphics Card drivers and CUDA 
# Note: To enable GPU support, Ensure CUDA and NVIDIA drivers are installed on your host machine

#2. download and install the latest  Anaconda3

#3. install python via conda
#conda install -c anaconda python=3.10

#4. isntall torch and cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 

#5. install dependance
pip install -r requirements.txt

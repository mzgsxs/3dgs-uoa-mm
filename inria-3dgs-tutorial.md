# Step-by-step tutorial for getting 3dgs running

Make sure you have installed Nvidia driver (version > 530), CUDA (version > 12) and Docker. It may be a lot easier to just use Windows with WSL, as there is an existing [Pre-built Windows Binaries](https://github.com/graphdeco-inria/gaussian-splatting/tree/main?tab=readme-ov-file#pre-built-windows-binaries) so you don't have to build it yourself. But you can still follow this if you are using ubuntu 22.04 (or anyother docker compatible linux distro)

Either pull `gaetanlandreau/3d-gaussian-splatting` from you GUI docker app in Windows or run the following command in your terminal:

````
docker pull gaetanlandreau/3d-gaussian-splatting:latest
````

Once you have this image pulled, clone 3dgs code into your current working directory:

````
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
````

Start your docker in nvidia-runtime mode with their code mounted in container:

````
docker run --rm --runtime=nvidia --gpus all -v ./gaussian-splatting:/root/gaussian-splatting -it sha256:ac7c0eedae703a2d8c1b366152d766b57c924887bfffdcc178c676162d580dc4 bash
````

You will need to update this docker's "diff-gaussian-rasterization" "simple-knn" python package, those are submodules of the original 3dgs project. 

````
cd /root/gaussian-splatting
pip install -q ./submodules/diff-gaussian-rasterization
pip install -q ./submodules/simple-knn
````

Once those packages are installed, download the data and train the model, it took ~15 mins with my RTX 3080ti

````
wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip
python train.py -s ./tandt/train
````

To see your rendering, inside windows, download and decompress their [viwer](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) In powershell, cd into `PATH_TO_YOUR_DOWNLOAD_FOLDER\viewers\bin` and run something like:

````
.\SIBR_gaussianViewer_app.exe -m \\wsl.localhost\Ubuntu-22.04\home\<YOUR_USER_NAME>\gaussian-splatting\output\<run-id>
````

Render the view from training data'position and orentation from your trained gaussians

````
python render.py -m output/<run-id>/
````

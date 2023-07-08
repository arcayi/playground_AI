# git config --global user.name arcayi
# git config --global user.email arcayi@qq.com
# git config --global --add safe.directory /workspaces/roadai/.devcontainer

sudo pip config set --global global.index-url http://pypi.mirrors.ustc.edu.cn/simple/
sudo pip config set --global global.extra-index-url http://192.168.1.10:7104/test/pypi/
sudo pip config set --global install.trusted-host "pypi.mirrors.ustc.edu.cn 192.168.1.10"

# sudo apt -y install --no-install-recommends \
#   sysstat build-essential gdb libdc1394-25 libgraphviz-dev

# sudo apt remove -y cmake
# sudo pip install -U cmake pip

# sudo apt -y --fix-broken install

# sudo apt remove --auto-remove -y libopencv-dev libopencv-contrib-dev
# SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
# # bash ${SCRIPT_DIR}/mount_data.sh
sudo apt -y install -f --no-install-recommends /workspaces/thirdparty/opencv-dist/py3.10-ubuntu22.04-cu1211/OpenCV-4.8.0-x86_64-*.deb

# sudo pip install -r ${SCRIPT_DIR}/requirements.txt
# # sudo pip install fastdeploy-gpu-python==1.0.4 -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html

# pip uninstall -y opencv-python
# sudo pip uninstall -y opencv-python

## 安装docker
# sudo apt-get update
# sudo apt-get install -y ca-certificates curl gnupg lsb-release
# sudo mkdir -p /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
# echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
#   $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
# sudo chmod a+r /etc/apt/keyrings/docker.gpg
# sudo apt-get update
# sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
# sudo apt -y install --no-install-recommends docker-ce-cli docker-compose-plugin docker-buildx-plugin

USER=vscode
sudo usermod -aG docker $USER

unset PIP_CACHE_DIR

echo "source /workspaces/roadai/.devcontainer/env.sh" >>/home/vscode/.bashrc

sudo apt-get install -y --no-install-recommends xorriso isolinux
sudo apt-get install -y --no-install-recommends python3-tk
# sudo apt-get install -y --no-install-recommends libssl1.1
sudo apt-get install -y --no-install-recommends libssl3

# sudo apt-get install -y --no-install-recommends cuda-11-8 nvidia-compute-utils-530 nvidia-utils-530

# for polygraphy
sudo pip install /workspaces/thirdparty/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp310-none-linux_x86_64.whl

# mim install mmengine
# # mim install "mmcv>=2.0.0rc4"
# pip install "mmcv>=2.0.0rc4" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
# mim install "mmdet>=3.0.0rc0"
# mim install "mmpose>=1.0.0rc0"

# pip install \
#   -e /workspaces/sportai.py/ai-models/ \
#   -e /workspaces/sportai.py/video_grabber/
# pip install \
#   -e /workspaces/sportai.py/bpose/ \
#   -e /workspaces/sportai.py/camera_calibrator \
#   -e /workspaces/sportai.py/sportai_py_msg/
# pip install \
#   -e /workspaces/sportai.py/sportai_py_poseclassifier \
#   -e /workspaces/sportai.py/sportai_py_posedistance \
#   -e /workspaces/sportai.py/sportai_sprint/ \
#   -e /workspaces/sportai.py/sportai_faceid/ \
#   -e /workspaces/sportai.py/sportai_ballthrowing
# pip install \
#   -e /workspaces/sportai.py/sportai_py/

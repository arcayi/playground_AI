export TWINE_USERNAME=test
export TWINE_PASSWORD=test
export PATH=$PATH:/workspaces/playground_AI/.devcontainer:/workspaces/playground_AI/.devcontainer/script:

# docker from docker
sudo chmod 666 /var/run/docker.sock

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11/lib64/

# pipenv
# eval "$(_PIPENV_COMPLETE=bash_source pipenv)"

# pyenv
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# export LD_LIBRARY_PATH=/opt/fastdeploy/lib:/opt/fastdeploy/third_libs/install/onnxruntime/lib:/opt/fastdeploy/third_libs/install/paddle2onnx/lib:/opt/fastdeploy/third_libs/install/tensorrt/lib:/opt/fastdeploy/third_libs/install/paddle_inference/paddle/lib:/opt/fastdeploy/third_libs/install/paddle_inference/third_party/install/mkldnn/lib:/opt/fastdeploy/third_libs/install/paddle_inference/third_party/install/mklml/lib:/opt/fastdeploy/third_libs/install/openvino/runtime/lib:/opt/fastdeploy/third_libs/install/opencv/lib64/:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.1/NsightSystems-cli-2023.2.1/host-linux-x64

export GIT_TERMINAL_PROMPT=1
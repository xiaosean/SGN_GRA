FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04

#get deps
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev

RUN apt-get install -y python3-setuptools
RUN apt-get install git

#for python api
RUN pip3 install wheel scikit-build numpy h5py
RUN pip3 install cython sklearn

RUN pip3 install torch torchvision pandas matplotlib

# RUN pip3 install tqdm
# RUN pip3 install torchlars
# RUN pip3 install tensorboradX

CMD ["python3"]
FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update && apt-get install -y apt-file && apt upgrade -y
RUN apt install -y build-essential curl git cmake pkg-config checkinstall 
RUN apt-get update && apt-get -y install sudo

RUN apt-get update && apt install -y gfortran libjpeg8-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev 
RUN apt install -y libxine2-dev libv4l-dev liblmdb-dev libleveldb-dev libsnappy-dev mesa-utils libgl1-mesa-glx freeglut3-dev

RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so



RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3
RUN conda install python=3.8.13

RUN conda update -y -n base -c defaults conda
RUN conda install -y pip
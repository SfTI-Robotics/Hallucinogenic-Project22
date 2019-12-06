# Base Tensorflow image
FROM tensorflow/tensorflow:latest-gpu-py3 

WORKDIR /home

COPY requirements.txt /home/requirements.txt

RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Installing python dependencies defined in requirements.txt
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Define environment variable
ENV NAME RwangMratP4p

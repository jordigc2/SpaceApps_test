# set base image (host OS)
#FROM python:3.7.1
FROM jjanzic/docker-python3-opencv

# set the working directory in the container
WORKDIR /spaceapps_test

RUN pip install --upgrade pip

#Install required packages
RUN pip3 install numpy scikit-learn matplotlib tensorflow segmentation-models albumentations
#keras-segmentation
RUN pip3 install jupyter

#run the container in bash
#ENTRYPOINT ["/bin/bash"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

#sudo docker run -p 8888:8888 -v /home/jgc/Documents/SpaceApps_test/src/:/spaceapps_test/src/ -v /home/jgc/Documents/SpaceApps_test/data/:/spaceapps_test/data/ spaceapps


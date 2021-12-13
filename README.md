# SpaceApps_test
Programming test to perform Semantic Segmentation from the data set AI4Mars

## Preparing data
Create the folder **./data** and store inside the AI4Mars dataset provided in google drive.

## Building image
```
sudo docker build -t spaceapps .
```

## Running container
```
sudo docker run -p 8888:8888 -v <path-from-root-to-repo>/SpaceApps_test:/spaceapps_test/ spaceapps
```

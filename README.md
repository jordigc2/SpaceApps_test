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

## Running code
Go into the folder src and open the **Code.ipynb** to see the last execution done with all the description. You should be able to run the different cells.

### NOTE
I wanted to upload the NN weiths but the .zip file was 113MB, which made it not possible to upload to github. Use [this link](https://drive.google.com/file/d/1Nvc611AhQ2IUn9lalt-5-86Asrni8-ri/view?usp=sharing) to Drive to download the checkpoints and copy them inside the **./src** folder.


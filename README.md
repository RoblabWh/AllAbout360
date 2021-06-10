# Introduction
The AllAbout360 Software tries to provide an easy way to calibrate different cameras with little effort. For this you will need to use the calibration UI, which will generate the mappingiles needed for the mappingsoftware. The Mappingsoftware is used to create the equirectengularprojection and also provides multiple usemodes for different cases.

# Dependencies
## required
Opencv

## optional
OpenMP

Cuda

Opencl

# Installation
The Installation only needs a basic gitclone followed by the cmake command
```
git clone https://github.com/RoblabWh/AllAbout360-.git
cd cd AllAbout360-/
mkdir build
cd build
cmake ..
make
```
# Calibration
The Calibration part is used to create a mappingfile that fits your need. You can create Mappingfiles for different Resolutions or to fix rotated cameras etc.


## Starting Calibration
start by running the example provided within the Project or with your own videofile
```
./dfe2eqr-calibration ../example.mp4
```
This should spawn 3 windows, one for the projection, and one window for each eye. Each Window has unique Parameters, this means the two fisheye parameters are independend.
## Equirectangular preview
![Equiwindow](https://user-images.githubusercontent.com/74601419/119018559-7aba5280-b99c-11eb-961e-bacb8ad88f91.png)
## fisheye 1
![fisheye1](https://user-images.githubusercontent.com/74601419/119018634-958cc700-b99c-11eb-8223-6d8cdb25af04.png)
## fisheye 2
![fisheye2](https://user-images.githubusercontent.com/74601419/119018642-99204e00-b99c-11eb-8b2c-6251f6eb5056.png)

## Parameters
### offset x
Horizontal offset in Pixel
### offset y
Vertical offset in Pixel
### radius
Radius of fisheye Image in pixel
### field of view
Field of view of the camera, Radius and fov are not independend!! That means multiple radius and fov settings could result in the same output.
### rotation
Rotation of the camera

## Example calibration
Fist off we have to think about the range in which we want to archive the highest precision, since the projection won't fit for all distances.
We let the video play until we find some nice edges that we can try to match.
![Equiedges](https://user-images.githubusercontent.com/74601419/119023831-b2c49400-b9a2-11eb-9163-2fd47c9ea48c.png)

x and y parameters are rarely usefull and are better ignored for a fast calibration.

fov can be set to the fov of the camera, our default is 190 degree and that is also our camera fov so we dont have to change that.

Now we can start to use Radius. It is visible with the green/red ring in the fisheye windows, but we find it easier to use the previewscreen while playing with the value to match the edges. You should try to keep the Radius on both eyes equal.
![equiradius](https://user-images.githubusercontent.com/74601419/119026259-3ed7bb00-b9a5-11eb-9ef2-b054bb55fb12.png)

In this picture were holding the camere, so the rotation need to be used. It is important to remember, that one fisheye might be tilted by 1 degree to the right, which will cause the other fisheye to mirror that change and tilt 1 degree to the left.

![Equirot](https://user-images.githubusercontent.com/74601419/119027264-6ed38e00-b9a6-11eb-80f8-14902105fb85.png)

At this step you might need to try some configurations with different radius and rotations. A small error on two sides normally results in a better image then one good and one bad edge.
The last Parameter would be the blend. You can also use the blend to see how your eyes overlap and use that to make a better calibration.

![blendimage](https://user-images.githubusercontent.com/74601419/119028611-f5d53600-b9a7-11eb-917e-96a454249a67.png)

To genarate the Mappingfile press enter. It will be generated as mapping-table.txt in the build folder.

# Mapping
The Mappingsoftware uses a mappingfile and one or two input videofiles to create a equirectangularprojection.

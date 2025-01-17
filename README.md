# Face Recognition with InsightFace
Recognize and manipulate faces with Python and its support libraries.  
The project uses [MTCNN](https://github.com/ipazc/mtcnn) for detecting faces, then applies a simple alignment for each detected face and feeds those aligned faces into embeddings model provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, a softmax classifier and cosinSimilarity were put on top of embedded vectors for classification task.

## Getting started
### Requirements
- Python 3.3+
- Virtualenv
- python-pip
- mx-net
- tensorflow
- Linux
### Installing 
Check update:
```
sudo apt-get update
```
Install python:
```
sudo apt-get install python3.6
```
Install pip:
```
sudo apt install python3-pip
```
Most of the necessary libraries were installed and stored in `env/` folder, so what we need is installing `virtualenv` to use this enviroment.  
Install virtualenv:
```
sudo pip3 install virtualenv virtualenvwrapper
```
## Usage
First, go to directory that you have cloned, activate __env__ to use installed package, alternatively, you must install all packages that necessary for this project.
```
source env/bin/activate
```
Now move to __/src__ folder and try out first recognition with this command:
```
python3 recognizer_image.py 
```
Real time run:
```
python3 recognizer_stream.py
```
## Build your own faces recognition system
By default, most of the input and output arguments were provided, models and embeddings is set default stored in `/src/outputs/`.  
### 1. Prepare your data 
Our training datasets were built as following structure:
```
/datasets
  /train
    /person1
      + face_01.jpg
      + face_02.jpg
      + ...
    /person2
      + face_01.jpg
      + face_02.jpg
      + ...
    / ...
  /test
  /unlabeled_faces
  /videos_input
  /videos_output
```
In each `/person_x` folder, put your face images corresponding to _person_name_ that has been resized to _112x112_ (input size for InsightFace). Here I provided two ways to get faces data from your webcam and video stored in your storage.  
__a. Get faces from camera__  
Run following command, with `--faces` defines how many faces you want to get, _default_ is 20
```
python3 get_faces_from_camera.py [--faces 'num_faces'] [--output 'path/to/output/folder']
```
Here `[--cmd]` means _cmd_ is optional, if not provide, script will run with its default settings.  
__b. Get faces from video__  
Prepare a video that contains face of the person you want to get and give the path to it to `--video` argument:
```
python3 get_faces_from_video.py [--video 'path/to/input/video'] [--output 'path/to/output/folder']
``` 
As I don't provide stop condition to this script, so that you can get as many faces as you want, you can also press __q__ button to stop the process.</br>
__c. Create face from `/data`__
```
/data
  /person1
    + img_01.jpg
    + img_02.jpg
    + ...
  /person2
    + img_01.jpg
    + img_02.jpg
    + ...
  / ...
```
You also put into `data/person_name/` image free size has only one face. Then run command:
```
python3 create_dataset.py
```
Model MTCNN will be detected face, cropped and put into  `dataset/train/person_name/`
The default output folder is `/unlabeled_faces`, select all faces that match the person you want, and copy them to `person_name` folder in `train`. Do the same things for others person to build your favorite datasets.
### 2. Download pre-trained models
Visited [here](https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0) download models pre-trained(include 3 file log, model-0000.params, models-symbols.json) and put into `/insightface/models/model-y1-test1`
### 3. Generate face embeddings
```
python3 faces_embedding.py [--dataset 'path/to/train/dataset'] [--output 'path/to/out/put/model']
```
### 4. Train classifier with softmax
```
python3 train_softmax.py [--embeddings 'path/to/embeddings/file'] [--model 'path/to/output/classifier_model'] [--le 'path/to/output/label_encoder']
```

### 5. Run
Yep!! Now you have a trained model, let's enjoy it!  
Face recognization with image as input:
```
python3 recognizer_image.py [--image-in 'path/to/test/image'] [...]
```
Face recognization with video as input:
```
python3 recognizer_video.py [--video 'path/to/test/video'] [...]
```
Face recognization with camera:
```
python3 recognizer_stream.py
```
`[...]` means other arguments, I don't provide it here, you can look up in the script at arguments part
## Others
### Using gpu for better performance
I use __CPU__ for all recognition tasks for __mxnet__ haven't supported for __cuda__ in Ubuntu 18.10 yet. But if your machine has an Nvidia GPU and earlier version of Ubuntu, you can try it out for better performance both in speed and accuracy.
In my case, I have changed __line 46__ in _face_model_ `ctx = mx.cpu(0)` to use cpu. 
### Thanks
- Many thanks to [Davis King](https://github.com/davisking) for creating dlib with lots of helpful function in face deteting, tracking and recognizing
- Thanks to everyone who works on all the awesome Python data science libraries like numpy, scipy, scikit-image, pillow, etc, etc that makes this kind of stuff so easy and fun in Python.
- Thanks to Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/) for their InsightFace project
- Thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) for his useful tutorials in [pyimagesearch](https://www.pyimagesearch.com/) that help me a lots in building this project.

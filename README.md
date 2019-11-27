# Face Recognition with MTCNN(face detect) + InsightFace(recognize)
Recognize and manipulate faces with Python and its support libraries.  
Bài toán sử dụng [MTCNN](https://github.com/ipazc/mtcnn) để detecting khuôn mặt, sau đó là tinh chỉnh khuôn mặt(xoay về đúng hướng) trước khi embedding face bằng model insightFace (vector 512D)[InsightFace](https://github.com/deepinsight/insightface). Cuối cùng, Neuron Network(softmax) + cosinSimilarity để dự đoán khuôn mặt thuộc class nào hoặc unknow

## Getting started
### Requirements
- Python 3.3+
- Virtualenv
- python-pip
- mx-net
- tensorflow
- Linux
### Installing 
Kiểm tra update:
```
sudo apt-get update
```
Cài đặt python:
```
sudo apt-get install python3.6
```
Cài đặt pip:
```
sudo apt install python3-pip
```
Cài đặt môi trường ảo.  
Install virtualenv:
```
sudo pip3 install virtualenv virtualenvwrapper
```
## Usage
Chạy môi trường ảo: vào thư mục đã clone và chạy:
```
source env/bin/activate
```
Vào thư mục __/src__ và chạy lệnh sau để nhận diện bằng ảnh:
```
python3 recognizer_image.py 
```
Nhận diện bằng camera:
```
python3 recognizer_stream.py
```

## Build your own faces recognition system
Mặc định, các file model và embedding face được lưu trong `/src/outputs/`.  
### 1. Prepare your data 
Tổ chức thư mục:
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
Mỗi thư mục `/person_x`, chứa ảnh khuôn mặt đã được cắt với tên thư mục _person_name_ và được được resize về kích thước 112 x 112 (kích thước đầu vào của InsightFace). Sau đây là các cách để tạo .  
__b. Get faces from camera__  
Run following command, with `--faces` defines how many faces you want to get, _default_ is 20
```
python3 get_faces_from_camera.py [--faces 'num_faces'] [--output 'path/to/output/folder']
```
Here `[--cmd]` means _cmd_ is optional, if not provide, script will run with its default settings.  
__c. Get faces from video__  
Prepare a video that contains face of the person you want to get and give the path to it to `--video` argument:
```
python3 get_faces_from_video.py [--video 'path/to/input/video'] [--output 'path/to/output/folder']
``` 
As I don't provide stop condition to this script, so that you can get as many faces as you want, you can also press __q__ button to stop the process.</br>
  
The default output folder is `/unlabeled_faces`, select all faces that match the person you want, and copy them to `person_name` folder in `train`. Do the same things for others person to build your favorite datasets.
### 2. Generate face embeddings
```
python3 faces_embedding.py [--dataset 'path/to/train/dataset'] [--output 'path/to/out/put/model']
```
### 3. Train classifier with softmax
```
python3 train_softmax.py [--embeddings 'path/to/embeddings/file'] [--model 'path/to/output/classifier_model'] [--le 'path/to/output/label_encoder']
```

### 4. Run
Yep!! Now you have a trained model, let's enjoy it!  
Face recognization with image as input (nhận diện ảnh):
```
python3 recognizer_image.py [--image-in 'path/to/test/image'] [...]
```
Face recognization with video as input (nhận diện trong video):
```
python3 recognizer_video.py [--video 'path/to/test/video'] [...]
```
Face recognization with camera (realtime):
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

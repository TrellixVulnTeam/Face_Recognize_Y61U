import os
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

def extract_face(filename, required_size=(112, 112)):
    image = cv2.imread(filename)
    # pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(image)
    print(results)
    if(len(results) == 0):
        return None
    # print(results)cd 
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    # x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    print(x1, x2, y1, y2)
    # extract the face
    face = image[y1:y2, x1:x2]
    # cv2.imshow('frame', face)
    # cv2.waitKey(0)
    face = cv2.resize(face, required_size)
    return face
# extract_face('/home/hung-vt/Face-Recognition-with-InsightFace/data/Hue/74209237_1002445520093975_5287619871163547648_n.jpg', (112, 112))
data_path = '/home/hung-vt/Face-Recognition-with-InsightFace/data'
dataset_path = '/home/hung-vt/Face-Recognition-with-InsightFace/datasets/train/'
out_put_data = '/home/hung-vt/Face-Recognition-with-InsightFace/datasets/train/'
people_data = os.listdir(data_path)
people_dataset = os.listdir(dataset_path)



# for person in people:
#     os.makedirs(os.path.join(out_put_data, person))
for person in people_data:
    if person not in people_dataset:
        print(person)
        os.makedirs(os.path.join(out_put_data, person))
        path_img = os.path.join(data_path, person)
        list_img = os.listdir(path_img)
        for img in list_img:
            file_img = os.path.join(path_img, img)
            face = extract_face(file_img, (112, 112))
            if face is None:
                continue
            cv2.imwrite(out_put_data + person + '/' + img, face)
    else:
        continue


# print(people)
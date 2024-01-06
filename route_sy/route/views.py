import re

from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse

import base64
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os


#def index(request):
 #   if request.session.get('login')==None:
 #       return redirect('/route/login/')
 #   return render(request,'html/index.html')



# views.py

from django.shortcuts import render,HttpResponse
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.conf import settings
import json

import json
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
import collections
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model

word_to_index = {}
with open ("C:/Users/Lenovo/Flickr30k-Image-Caption-Generator/data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open ("C:/Users/Lenovo/Flickr30k-Image-Caption-Generator/data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)



print("Loading the model...")
model = load_model('C:/Users/Lenovo/Flickr30k-Image-Caption-Generator/model_checkpoints/model_14.h5')

resnet50_model = ResNet50 (weights = 'E:/模型/resnet50_weights_tf_dim_ordering_tf_kernels(1).h5', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)

def index(request):
    return render(request,"index.html")

def predict_caption(photo):

    inp_text = "startseq"

    #max_len = 80 which is maximum length of caption
    for i in range(80):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')

        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]

        inp_text += (' ' + word)

        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def preprocess_image (img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img

def encode_image (img):
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    # feature_vector = feature_vector.reshape((-1,))
    return feature_vector

def runModel(img_name):
    #img_name = input("enter the image name to generate:\t")

    print("Encoding the image ...")
    photo = encode_image(img_name).reshape((1, 2048))

    print("Running model to generate the caption...")
    caption = predict_caption(photo)

    return caption

def files(request):

    # 由前端指定的name获取到图片数据
    img = request.FILES.get('img')
    print(type(img))
    # 获取图片的全文件名
    img_name = img.name
    # 截取文件后缀和文件名
    mobile = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]
    # 重定义文件名
    img_name = f'{mobile}{ext}'
    # 从配置文件中载入图片保存路径
    img_path = os.path.join(settings.IMG_UPLOAD, '1.jpg')
    # 写入文件

    os.remove(img_path)

    with open(img_path, 'ab') as fp:
        # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
        for chunk in img.chunks():
            fp.write(chunk)
    caption=runModel(img_path)
    print(caption)

    return render(request, 'index2.html', {'response': caption})
    #return render(request,"index.html")


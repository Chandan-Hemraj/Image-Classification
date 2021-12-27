import os
import boto3
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR , 'model.hdf5'))
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])


ACCESS_KEY = 'AKIA5YVCEFAAOGN7HUX5'
SECRET_KEY = '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'

s3 = boto3.client('s3',
                  aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY,
                  )

BUCKET_NAME = 'imageclassification'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']

def predict(filename , model):
    img = load_img(filename , target_size = (32 , 32))
    img = img_to_array(img)
    img = img.reshape(1 , 32 ,32 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result

def start():
    
    img_path = 'images/image.jpg'
    try:
        KEY = sys.argv[1]
        s3.download_file(BUCKET_NAME, KEY, KEY)
        
    except:    
        KEY = 'plane.jpg'
        print("\nDefault image taken : plane.jpg\n")
        print("please pass the argumnet as image\n")
        s3.download_file(BUCKET_NAME, KEY, KEY)

    class_result , prob_result = predict(KEY , model)

    #s3.delete_object(Bucket=BUCKET_NAME, Key=KEY)

    predictions = {
                    class_result[0]: prob_result[0],
                    class_result[1]: prob_result[1],
                    class_result[2]: prob_result[2],
                }

    print(predictions)
    print("\n")
    os.remove(KEY)
    return predictions


start()

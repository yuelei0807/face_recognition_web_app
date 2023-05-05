import numpy as np
import sklearn
import pickle
import cv2

#Load required models
#Load the cascade classifier
cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
#Load the support vector machine model
svm = pickle.load(open('./model/support_vector_machine_model.pickle',mode='rb'))
#Load the PCA dictionary
pca_models = pickle.load(open('./model/pca_dictionary.pickle',mode='rb'))
pca = pca_models['pca']
mean_face_array = pca_models['mean_face']

def faceRecognitionPipeline(filepath,path=True):
    #create pipeline
    if path:
        #Read images
        img = cv2.imread(filepath)
    else:
        #array
        img = filepath
       
    #Convert images into grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Crop faces using the cascade classifier
    crop_faces = cascade.detectMultiScale(gray,1.5,3)
    predictions = []
    #draw the bounding boxes
    for x,y,w,h in crop_faces:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        region = gray[y:y+h,x:x+w]
        #normalize images
        region = region / 255.0

        #resize images
        if region.shape[1] > 100:
            region_resize = cv2.resize(region,(100,100),cv2.INTER_AREA)
        else:
            region_resize = cv2.resize(region,(100,100),cv2.INTER_CUBIC)
        region_reshape = region_resize.reshape(1,10000)

        #subtract cropped faces with mean
        region_mean = region_reshape - mean_face_array
        #get Eigen image by applying mean face to PCA model
        eigen_image = pca.transform(region_mean)
        #visulize Eigen image
        eigen_img = pca.inverse_transform(eigen_image)
        #pass Eigen image into svm model and get predictions
        results = svm.predict(eigen_image)
        score = svm.predict_proba(eigen_image)
        max_score = score.max()
        #generate report
        output = "%s : %.2f%%"%(results[0],max_score*100)
        #print(output)
        #Show predictions results on the original image, mark female in red box, and male in blue box.
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color = (255,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,output,(x,y),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),5)
        prediction = {
            'cropped_face': region,
            'eigen_image': eigen_img,
            'prediction': results[0],
            'probability': max_score
        }
        predictions.append(prediction)
    return img, predictions
     

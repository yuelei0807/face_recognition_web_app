import os
import cv2
import matplotlib.image as matimg
from app.face_recognition import faceRecognitionPipeline
from flask import render_template, request

upload_folder = 'static/upload'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def gender_prediction():
    if request.method == 'POST':
        file = request.files['image_name']
        filename = file.filename
        #save the image uploaded into the upload folder
        path = os.path.join(upload_folder,filename)
        file.save(path)
        #out the prediction result
        img_predicted, predictions = faceRecognitionPipeline(path)
        predictionImageSave = 'prediction_result.jpg'
        cv2.imwrite(f'./static/predict/{predictionImageSave}',img_predicted)
        
        #output the result report
        result_report = []
        for i, face in enumerate(predictions):
            gray_cropped_face = face['cropped_face']
            Eigen_image = face['eigen_image'].reshape(100,100)
            gender_prediction = face['prediction']
            probability = face['probability']

            #save the grayscale image and Eigen image into the predict folder
            cropped_face_name = f'cropped_face_{i}.jpg'
            eigen_image_name = f'eigen_image_{i}.jpg'
            matimg.imsave(f'./static/predict/{cropped_face_name}',gray_cropped_face,cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}',Eigen_image,cmap='gray')

            #save the output to the output list result_report[]
            result_report.append([cropped_face_name,eigen_image_name,gender_prediction,probability])
            #POST request
        return render_template('gender_prediction.html',imageupload=True,report=result_report)

    #GET request
    return render_template('gender_prediction.html',imageupload=False)


from imutils.object_detection import non_max_suppression 
import numpy as np 
import imutils 
import cv2 
import requests 
import time 
import argparse 
import base64

URL_EDUCATIONAL = "http://things.ubidots.com"
URL_INDUSTRIAL = "http://industrial.api.ubidots.com"
INDUSTRIAL_USER = False  
TOKEN = "BBFF-BWA4EGpRxLinoaM4HsBq3PkUNOdbZ9"  
DEVICE = "Bus_1"  
VARIABLE = "Passenger"  
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def localDetect(image_path):
    result = []
    image = cv2.imread(image_path)

    if len(image) <= 0:
        print("[ERROR] could not read your local image")
        return result
    print("[INFO] Detecting people")
    result = detector(image)

    for (x, y, w, h) in result:
    	cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (result, image)

def cameraDetect(token, device, variable, sample_time=5):

    cap = cv2.VideoCapture(0)
    init = time.time()

    if sample_time < 1:
        sample_time = 1

    while(True):
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        result = detector(frame.copy())

        for (x, y, w, h) in result:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('frame', frame)

        if time.time() - init >= sample_time:
            print("[INFO] Sending actual frame results")
            b64 = convert_to_base64(frame)
            context = {"image": b64, "Suggestion" : suggest(len(result)), "Availability" : "Bus_2 Availalble in 20 minutes"}
	    
            sendToUbidots(token, device, variable,
                          len(result), context=context)
            init = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def suggest(value):
    if value<50:
	count=50-value
        return "Available seats:"+str(count)   
    else:
	return "Seats Unvailable"

def convert_to_base64(image):
    image = imutils.resize(image, width=400)
    img_str = cv2.imencode('.png', image)[1].tostring()
    b64 = base64.b64encode(img_str)
    return b64.decode('utf-8')

def detectPeople(args):
    image_path = args["image"]
    camera = True if str(args["camera"]) == 'true' else False

    if image_path != None and not camera:
        print("[INFO] Image path provided, attempting to read image")
        (result, image) = localDetect(image_path)
        print("[INFO] sending results")
        b64 = convert_to_base64(image)
        context = {"image": b64}

        req = sendToUbidots(TOKEN, DEVICE, VARIABLE,
                            len(result))
        if req.status_code >= 400:
            print("[ERROR] Could not send data to Ubidots")
            return req

    if camera:
        print("[INFO] reading camera images")
        cameraDetect(TOKEN, DEVICE, VARIABLE)

def buildPayload(variable, value, context):
    return {variable: {"value": value, "context": context}}


def sendToUbidots(token, device, variable, value, context={}, industrial=True):
    url = URL_INDUSTRIAL if industrial else URL_EDUCATIONAL
    url = "{}/api/v1.6/devices/{}".format(url, device)

    payload = buildPayload(variable, value, context)
    headers = {"X-Auth-Token": token, "Content-Type": "application/json"}

    attempts = 0
    status = 400

    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)
        return req

def argsParser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", default=None)
    ap.add_argument("-c", "--camera", default=True)
    args = vars(ap.parse_args())

    return args

def main():
    args = argsParser()
    detectPeople(args)


if __name__ == '__main__':
    main()

import numpy as np
import datetime
import time
import cv2


def averageWeight(weightData):
    sum = 0
    for i in range(len(weightData)):
        average = np.average(weightData[i])
        sum += average
        print(f'Berat rata-rata video ke-{i+1} adalah {average}') 
    return average


def calculateWeight(age_list, ip_camera, percent, n_data, weightData, predictor):
    for i in range(len(ip_camera)):
        age = age_list[i]
        cap = cv2.VideoCapture(ip_camera[i])
        print(f'======== Memulai perhitungan berat video ke-{i+1} ========')
        try:
            while(cap.isOpened()):

                for j in range(n_data):
                    ret, im = cap.read()

                    if (ret == True) :
                        outputs = predictor(im, age, percent)
                        assert outputs["weight"] > -1
                        print(f'perhitungan berat pada pengambilan ke-{j+1}  adalah {outputs["weight"]}')
                        weightData[i][j] = outputs["weight"]
                        
                    time.sleep(1) # delay 10  detik untuk setiap pengambilan data
                break
        except:
            print(f"Tidak ada ayam terdeteksi pada video ke-{i+1}")
            #weightData.
            ayam_Detect = False
            continue        

    print("============ Memulai perhitungan berat rata-rata ============")
    total_average = averageWeight(weightData)
    return total_average

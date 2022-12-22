import datetime
import os
import numpy as np
import paho.mqtt.publish as publish

def unixToDatetime(ut):
    return datetime.datetime.fromtimestamp(ut)

def saveLocal(weightData,average,time):
    save_path = './SaveData/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        count = 0
    for path in os.scandir(save_path):
        if path.is_file():
            count += 1
    if count >1000:
        pass
    else:
        np.savetxt(save_path+'WeightData'+str(time.day)+str(time.month)+str(time.year)+'-'+str(time.hour)+str(time.minute)+'.csv',weightData, fmt='%10.5f', delimiter=',',header='data '+str(time),footer='berat rata-rata'+str(average))
    return

def sendWeight(cursor, total_average, datetimes, age):
    #fungsi Send ke database (MQTT):
    publish.single("soka/weight", "{payload: "+str(total_average)+"}", hostname="broker.emqx.io")
    
    #fungsi Send ke database (SQL):
    # cursor.execute("INSERT INTO ayam VALUES (datetimes, age, total_average)")
    cursor.execute("INSERT INTO ayam VALUES (?, ?, ?)", (datetimes, age, total_average))
    
    #show db
    rows = cursor.execute("SELECT tanggal, umur, berat FROM ayam").fetchall()
    print(rows)
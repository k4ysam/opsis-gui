import tomli
import pandas as pd
import numpy as np
import time
import cv2

idList = [1, 2, 3, 4, 5]

tomlFile = open("calib_files/camera/caliscope.toml", mode='rb')
tomlConfig = tomli.load(tomlFile)

calibData = pd.DataFrame(index=idList, columns=["cameraMatrix", "distCoeffs", "R", "T"])
calibData = calibData.replace(np.nan, None)

for id in idList:
    calibData.at[id, 'cameraMatrix'] = tomlConfig[f'cam_{id}']['matrix']
    calibData.at[id, 'distCoeffs'] = np.array(tomlConfig[f'cam_{id}']['distortions']).reshape((1,-1))
    calibData.at[id, 'R'] = cv2.Rodrigues(np.array(tomlConfig[f'cam_{id}']['rotation']))[0]
    calibData.at[id, 'T'] = np.array(tomlConfig[f'cam_{id}']['translation']).reshape((-1,1))*1000
    
calibData.reset_index(names='id').to_json(f'calib_files/camera/{time.strftime("%Y-%m-%d %H_%M_%S")}.json', orient='records', indent=3)
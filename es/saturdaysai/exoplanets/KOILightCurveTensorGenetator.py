import pandas as pd
import numpy as np
from pathlib import Path
from lightKurveApi.lightKurveClient import LightKurveClient 

class KOILightCurveTensorGenerator:
    
    DEFAULT_GLOBAL_LEN = 2049+1e-9 
    DEFAULT_LOCAL_LEN = 257+1e-9 
    DEFAULT_LOCAL_VIEW_WIDTH = 4
    DEFAULT_FOLD_MODE = 'fold'
    
    def __init__(self, 
                 source_file_name, 
                 destination_folder_path, 
                 global_tensor_len = DEFAULT_GLOBAL_LEN, 
                 local_tensor_len = DEFAULT_LOCAL_LEN, 
                 local_view_witdh = DEFAULT_LOCAL_VIEW_WIDTH, 
                 fold_mode = DEFAULT_FOLD_MODE):
        
        self.df = pd.read_csv(source_file_name)
        self.destination_folder_path = destination_folder_path
        self.global_tensor_len = int(global_tensor_len)
        self.local_tensor_len = int(local_tensor_len)
        self.local_view_witdh = int(local_view_witdh)
        self.fold_mode = fold_mode
        
    def getTensors(self, window=None):
        
        if window == None:  # If no window is given, retrieve the whole dataset
            df = self.df
        else:
            print(f"Window: {window}")
            df = self.df.loc[window[0]:window[1]]
        
        lkClient = LightKurveClient()
        list_tensors_x = []
        list_tensors_y = []
        list_tensors_z = []
        for index, row in df.iterrows():
            print(row)
            koi_label = row.koi_is_planet
            mission = row.mission
            koi_id = row.koi_id
            koi_name = row.koi_name
            koi_t0 = row.koi_time0bk
            duration = row.koi_duration
            period = row.koi_period
            
            list_koi_tensors_x = lkClient.getKOILightKurve(
                                     koi_id,
                                     koi_t0, 
                                     period, 
                                     duration, 
                                     self.global_tensor_len, 
                                     self.local_tensor_len, 
                                     self.local_view_witdh, 
                                     mission = mission, 
                                     fold_mode = self.fold_mode)
            
            print(f"Obtained {len(list_koi_tensors_x)} tensors.")
            list_koi_tensors_y = [koi_label] * len(list_koi_tensors_x)
            list_koi_tensors_z = [koi_name] * len(list_koi_tensors_x)
            list_tensors_x = list_tensors_x + list_koi_tensors_x
            list_tensors_y = list_tensors_y + list_koi_tensors_y
            list_tensors_z = list_tensors_z + list_koi_tensors_z
        
        return list_tensors_x, list_tensors_y, list_tensors_z
    
    def persist(self, x, y, z, suffix):
        path = Path(self.destination_folder_path)
        np.save(path / f"X_{self.fold_mode}_{suffix}.npy", x)
        np.save(path / f"Y_{self.fold_mode}_{suffix}.npy", y)
        np.save(path / f"Z_{self.fold_mode}_{suffix}.npy", z)
        
# Execute only if script run standalone (not imported)    
if __name__ == '__main__':
    import sys

    (script, source_file_name, destination_folder_path, window_init, window_end, fold_mode) = sys.argv
    window_init = int(window_init)
    window_end = int(window_end)
    tensorGenerator = KOILightCurveTensorGenerator(source_file_name, destination_folder_path, fold_mode = fold_mode)
    (x,y,z) = tensorGenerator.getTensors([int(window_init),int(window_end)])
    tensorGenerator.persist(x,y,z,f"{window_init:05d}_{window_end:05d}")
    
    
    
    
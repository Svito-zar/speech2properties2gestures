import numpy as np 
import datetime
old = np.load("../../../dataset/processed/numpy_arrays/train_n_val_Y_Phrase.npy")
new = np.load("../../../dataset/processed/numpy_arrays/Phrase_properties.npy").astype(np.float32)


# compare speakers
speakers_old = np.unique(old[:, 0], axis=0)
speakers_new = np.unique(new[:, 0], axis=0)

assert np.array_equal(speakers_old, speakers_new)

for speaker in speakers_old:
    speaker_data_old = old[np.argwhere(old[:, 0] == speaker)].squeeze()
    speaker_data_new = new[np.argwhere(new[:, 0] == speaker)].squeeze()

    print("\n" + "_"*80 + "\n" + f"Speaker {int(speaker)}")
    print("\t", speaker_data_old.shape, "->", speaker_data_new.shape)

    list_old = speaker_data_old.tolist()
    list_new = speaker_data_new.tolist()
            
    missing_rows = [row[1:] for row in list_old if row not in list_new]
    extra_rows = [row[1:] for row in list_new if row not in list_old]
   
    
    print(len(missing_rows), "MISSING ROWS:")
    for row in missing_rows:
        timestamp = str(datetime.timedelta(seconds=row[0]))
        print(timestamp, row[1:])
        
    print(len(extra_rows), "EXTRA ROWS:")
    for row in extra_rows:
        timestamp = str(datetime.timedelta(seconds=row[0]))
        print(timestamp, row[1:])
        
    print("\n"+"_"*80)


import pympi
curr_folder = "/home/tarask/Documents/Datasets/SaGa/Annotations/SaGA-V07"
curr_file = curr_folder + "/V07-SGRef.eaf"
elan = pympi.Elan.Eaf(file_path=curr_file)

transcriptions = elan.tiers['R.Speech_form'][0]

time_key = elan.timeslots

for key, (st_t, end_t, word, label) in transcriptions.items():
    if word is not None:
        if word != "" and word != " ":
            print(time_key[st_t]/1000, time_key[end_t]/1000, word)

print("Done!")
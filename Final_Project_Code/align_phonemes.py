import numpy as np
from shutil import rmtree
import json
import os
import soundfile as sf


phones = []
directory = './json'

if os.path.isdir('txt'):
		rmtree('txt')
os.mkdir('txt')
#os.remove('./json/.DS_Store')
# read in all files in .wav directory
if os.path.isdir(directory):
	json_dir = './json'
	json_files = os.listdir(json_dir)
	for file in json_files:
		json_file = "./json/" + file
		if json_file:
			with open(json_file, 'r') as f:
				 words = json.load(f)
			wav_file = "./wav/" + file.split('.json')[0] + ".wav"
			x, fs = sf.read(wav_file)
			samples = len(x)
			time = samples/fs
			print(time)
			file = open("./txt/" + file.split('.json')[0] + ".txt","w") 
			word_start = 0
			word_end = 0
			for word in words["words"]:
				next_word_start = word["start"]
				if (word_end - next_word_start) != 0:
					file.write(str(word_end)+" ")
					file.write(str(next_word_start)+" ")
					file.write("pau")
					file.write("\n")
				word_start = word["start"]
				word_end = word["end"]
				phone_time = word_start
				for phone in word["phones"]:
					phone_start = phone_time
					phone_duration = phone["duration"]
					phone = phone["phone"].split('_')[0]
					phones.append(phone)
					phone_end = phone_time + phone_duration
					phone_time = phone_end
					file.write(str(phone_start)+" ")
					file.write(str(phone_end)+" ")
					file.write(phone)
					file.write("\n")
			if phone_time < time:
				file.write(str(phone_time)+" ")
				file.write(str(time)+" ")
				file.write("pau")
			file.close()
else:
	print("Can't find .json file directory!")

print(set(phones))
print(len(set(phones)))

  

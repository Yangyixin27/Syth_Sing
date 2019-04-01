# coding: utf-8

import os
import pandas as pd
import numpy as np

# load the japanese phone dictionary 
def load_phone_dict(path):
	phone_dict = list()
	with open(path, 'r') as f:
		for line in f:
			phone_dict.append(line.strip())
	return phone_dict  


def phone_to_one_hot(phone, phone_dict):
	tmp = list(np.zeros(len(phone_dict)))
	if phone=='None':   
		return tmp
	else:
		tmp[phone_dict.index(phone)] = 1.0
		return tmp

def time_to_frame(time, window_size, step_size):
	# number_windows=(time-2*window_size)//step_size
	number_windows = np.floor(time/step_size)+1
	if number_windows >= 0:
		return number_windows
	else:
		return 0

def corase_encoding(current_index, beg, end):
	middle = int((end-beg)/2)
	if current_index == beg:
		return [1, 0, 0]
	elif current_index == middle:
		return [0, 1, 0]
	elif current_index == end:
		return [0, 0, 1]
	else:
		inverse_dist_beg = 1/np.abs(current_index-beg)
		inverse_dist_middle = 1/np.abs(current_index-middle)
		inverse_dist_end = 1/np.abs(current_index-end)
		p_beg = inverse_dist_beg/(inverse_dist_beg+inverse_dist_middle+inverse_dist_end)
		p_middle = inverse_dist_middle/(inverse_dist_beg+inverse_dist_middle+inverse_dist_end)
		p_end = inverse_dist_end/(inverse_dist_beg+inverse_dist_middle+inverse_dist_end)
		return [p_beg, p_middle, p_end]
def creat_folder(folder_name):
	try:  
		os.mkdir(folder_name);
		print('folder created: ', folder_name)
	except:
		print('folder already exist: ', folder_name)
		# pass
if __name__ == "__main__":

	#### convert /mono/.labs to csv
	#### a floder under /mono/ called /csv/ will be created and it will hold all the csv files with current, next, before
	#### the format of one example will be shown below

	label_path = '/Users/ShiyuMu/Desktop/HTS-demo_NIT-SONG070-F001/data/labels/mono/'
	fileList = [label_path + f for f in os.listdir(label_path) if f.endswith('.lab')]
	try:
		# create dictionary to hold all the csv file 
		os.mkdir(label_path + 'csv/');
		print('csv dictionary created')
	except:
		pass
	for file in fileList:
		df = pd.read_csv(file, sep=" ", header=None)
		df.columns = ["beg", "end", "current"]
		df['before'] = pd.Series(['None']).append(df['current']).reset_index(drop=True)
		df['next'] = df['current'][1:].append(pd.Series(['None'])).reset_index(drop=True)
		df.to_csv(label_path + 'csv/'+ file.split('/')[-1]+'.csv', sep='\t', index=False)

	### example of loading csv files 
	## this is the path we just created
	csv_path = label_path + 'csv/'
	# get all csv files 
	csvList = [csv_path + f for f in os.listdir(csv_path) if f.endswith('.csv')]
	df_list = list()
	## add all csv files as dataframe to a list
	for file in csvList:
		df = pd.read_csv(file, sep='\t')
		df_list.append(df)
	## the df_list contains 31 dataframe (since we have 31 dataset)
	## one dataframe will look like this. beg and end here indicates nonoseconds
	
	# df_list[0].head()


	#### load phoneme dictionary
	phone_dict_path = 'JPN_phone_dict.txt'
	phone_dict = load_phone_dict(phone_dict_path)

	### convert timestamps to frame numbers
	#### define window size and step size to convert timestamp to frame id
	#### define window size and step size to convert timestamp to frame id
	window_size = 250000 ## ingorned
	step_size = 50000

	for df in df_list:
		total_time = list(df['end'])[-1]
		total_num_frames = time_to_frame(total_time, window_size, step_size)
		
		for index in range(len(df['beg'].values)):
			df['beg'].values[index] = np.floor(df['beg'].values[index]/total_time*total_num_frames)
		for index in range(len(df['end'].values)):
			df['end'].values[index] = np.floor(df['end'].values[index]/total_time*total_num_frames)

	## now df_list contains 31 dataframe, one would be like this
	print(df_list[0].head())



	### Unfold all frames and list them one by one
	unfolded_df_list = list()
	for dataset in df_list:
		unfold_list = list()
		for index in range(dataset.shape[0]):
			beg_ = dataset.iloc[index]['beg']
			end_ = dataset.iloc[index]['end']
			current_ = np.asarray(phone_to_one_hot(dataset.iloc[index]['current'], phone_dict))
			before_ = np.asarray(phone_to_one_hot(dataset.iloc[index]['before'], phone_dict))
			next_ = np.asarray(phone_to_one_hot(dataset.iloc[index]['next'], phone_dict))
			corase_encoding_vectors = np.asarray(corase_encoding(index, beg_, end_))
	#     print(corase_encoding_vectors)
			if index == 0:
				unfold_list.append([beg_, end_, np.asarray(phone_to_one_hot('None', phone_dict)), current_, next_, corase_encoding(beg_, beg_, end_)])
			else:
				unfold_list.append([beg_, end_, before_, current_, next_, corase_encoding(beg_, beg_, end_)])
			for i in range(beg_+1, end_):
				unfold_list.append([beg_, end_, before_, current_, next_, corase_encoding(i, beg_, end_)])
			if index == (dataset.shape[0] -1):
				unfold_list.append([beg_, end_, before_, current_, np.asarray(phone_to_one_hot('None', phone_dict)), corase_encoding(end_, beg_, end_)])
			else:
				unfold_list.append([beg_, end_, before_, current_, next_, corase_encoding(end_, beg_, end_)])
		unfold_df = pd.DataFrame(unfold_list, columns=['beg', 'end', 'before', 'current', 'next', 'coarse'])
		print(unfold_df.shape)
		unfolded_df_list.append(unfold_df)

	## see how corase_encoding works. 
	## Input: current position, beg of this phoneme, end of this phoneme 
	## Output: the probability of the current frame's position in the [begining, middle, end] of the current phoneme
	# print(corase_encoding(110, 0, 233))
	## This function has been built in the feature generation process itself so you don't need to worry about it.

	## now unfolded_df_list is a list contains 31 dataframe, 
	## one example would be like this, the index at the front means frame id
	# print(unfolded_df_list[0]['end'])

	### Save npy files as required
	creat_folder('npy')  
	for i in range(len(csvList)):
		lab_name = csvList[i].split('/')[-1].split('.csv')[0]
		creat_folder('./npy/'+lab_name)
		np.save('./npy/'+lab_name+'/current.npy', unfolded_df_list[i]['current'])
		np.save('./npy/'+lab_name+'/next.npy', unfolded_df_list[i]['next'])
		np.save('./npy/'+lab_name+'/before.npy', unfolded_df_list[i]['before'])
		np.save('./npy/'+lab_name+'/corase_encoding.npy', unfolded_df_list[i]['coarse'])
		print('npy files saved for: ', lab_name)
	## Example of loading a noy file
	np_array_example = np.load('./npy/nitech_jp_song070_f001_010.lab/current.npy')
	## convert to shape (frame_number, features_number)
	np_array_example = np.array(np_array_example.tolist())
	print(np_array_example)
	# print(np_array_example)

	
	#### Since the requirement changed, the functions below are not useful anymore 
	# #### ### save the unfolded dataframe to disk

	# ### a folder under /mono/csv/ called /unfolded/ will becreated which holds all datafames like above
	# try:
	# 	# create dictionary to hold all the unfolded file 
	# 	os.mkdir(label_path + 'csv/unfolded/');
	# 	print('unfolded folder created')
	# except:
	# 	print('folder already exist')
	# 	pass
	# for i in range(len(csvList)):
	# 	unfolded_df_list[i].to_csv(label_path + 'csv/unfolded/'+ csvList[i].split('/')[-1], sep='\t', index=False)

	# ### After you created all the files, you can load them back to memory easily

	# ## NOTICE, if you already have all the files saved, 
	# ## the code below is the only thing you need to load the file and 
	# ## start to work with your training. The code before is only for generating. 

	# ## this is the path we just created
	# unfolded_path = label_path + 'csv/unfolded/'
	# # get all csv files 
	# fileNameList = [unfolded_path + f for f in os.listdir(unfolded_path) if f.endswith('.csv')]
	# unfolded_df_list_ = list()
	# ## add all csv files as dataframe to a list
	# for file in fileNameList:
	# 	df = pd.read_csv(file, sep='\t')
	# 	unfolded_df_list_.append(df)

	# # unfolded_df_list_[0].head()
	# ### of course you may also want the filename of this specify dataframe to match with raw files

	# # the index of fileNameList maintains the same with unfolded_df_list_
	# fileNameList[0].split('/')[-1]

	# ### get one-hot-vector, call phone_to_one_hot() to a specific phoneme
	# one_hot = phone_to_one_hot('sil', phone_dict)
	# print('sil', one_hot)

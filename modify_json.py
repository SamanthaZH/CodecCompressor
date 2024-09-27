import numpy as np
import json

### for lego 


test_set = ['train','test','val']
for test in test_set:
	split = test.split("_")[0]
	file = 'transforms_'+split+'.json'
	new_file = 'new_transforms_'+split+'.json'
	with open(file, 'r') as r_f:
		meta = json.load(r_f)
		camera_angle_x = meta['camera_angle_x']
		data=meta['frames']
		for i in range(len(data)):
			d = data[i]
			path = d['file_path'].split('/')[-1]
			new_path = './'+split+'/'+path
			data[i]['file_path'] = new_path

	# print(len(data))
	meta['camera_angle_x'] = camera_angle_x
	meta['frames'] = data

	with open(new_file, 'w') as w_f:
		json.dump(meta, w_f, indent=4)
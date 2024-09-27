import os
import subprocess
import sys

frame = 20
test_set = ['train_80','val_20']
for test in test_set:
	tot_f = int(test.split("_")[-1])
	split = test.split("_")[0]
	for i in range(tot_f):
		s_name = test+"/r_"+str(i)+"/"+str(frame).zfill(2)+".png"
		d_name= split+"/r_"+str(i)+".png"
		cmd = "mv "+s_name+" "+d_name
		subprocess.run(["mv",s_name,d_name])


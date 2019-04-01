import os
import numpy as np
import pdb

filenames = os.listdir("./phones/prev/")

mfsc_path = "./mfsc/"
ap_path = "./ap/"
f0_path = "./f0coded/"

dimensions = dict()

file = open("dimensions.txt","w") 
for i in filenames:
    prev_phone = np.array(np.load("./phones/prev/" + i).tolist())
    # prev_phone = np.array(prev_phone.tolist())

    cur_phone = np.array(np.load("./phones/current/" + i).tolist())
    next_phone = np.array(np.load("./phones/next/" + i).tolist())
    phone_pos = np.array(np.load("./phones/pos/" + i).tolist())

    mfsc = np.load(mfsc_path + i)
    ap = np.load(ap_path + i)
    f0 = np.load(f0_path + i)
    # pdb.set_trace()
    print ("Before, filename:{}, prev_phone:{}, phone_pos:{}, mfsc:{}, ap:{}, f0:{}".format(i, 
                                                                        prev_phone.shape, phone_pos.shape, mfsc.shape, ap.shape, f0.shape))
   
    file.write(i+" ")
    file.write(str(mfsc.shape[0]))
    file.write("\n")
   
    # if prev_phone.shape[0] != mfsc.shape[0]:
    # 	prev_phone = prev_phone[:-1, :]
    # 	cur_phone = cur_phone[:-1, :]
    # 	next_phone = next_phone[:-1, :]
    # 	phone_pos = phone_pos[:-1, :]

    # print ("After, filename:{}, prev_phone:{}, cur_phone:{}, next_phone:{}, phone_pos:{}, mfsc:{}, f0:{}".format(i, prev_phone.shape, cur_phone.shape, next_phone.shape, phone_pos.shape, mfsc.shape, f0.shape))

    np.save("./phones/prev/" + i, prev_phone)
    np.save("./phones/current/" + i, cur_phone)
    np.save("./phones/next/" + i, next_phone)
    np.save("./phones/pos/" + i, phone_pos)
    
    print ("Recreated npy for {}".format(i))
file.close()
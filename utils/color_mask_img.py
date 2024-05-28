import numpy as np
import cv2
import os
from misc import check_mkdir
img_path = ''
mask_path = ''
save_mask_path = ''
img_list= [img_path + f for f in os.listdir(img_path) if
               f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
gts_list = [mask_path + f for f in os.listdir(mask_path) if f.endswith('.jpg')
            or f.endswith('.png') or f.endswith('.bmp')]
images = sorted(img_list)
gts = sorted(gts_list)
merge_threshold = 0.5
merge_group = (0, 255, 0) #r,g,b
print(images)
check_mkdir(os.path.join(save_mask_path,str(merge_group[0])+'_'+str(merge_group[1])+'_'+str(merge_group[2])+'_'+str(merge_threshold)))
for i in range(len(images)):
    img_np = cv2.imread(images[i])
    _, image_name = os.path.split(images[i])
    mask_np_q = cv2.imread(gts[i])
    prediction_np_q = np.array(mask_np_q).astype(np.float)
    print(prediction_np_q.shape)
    img_np = np.array(img_np,dtype=float)


    img_np[prediction_np_q[:, :, 0 ] > 128] *= merge_threshold
    img_np[prediction_np_q[:, :, 0]  > 128,0]  += merge_group[2]*(1-merge_threshold)
    img_np[prediction_np_q[:, :, 0] > 128, 1]  +=  merge_group[1]*(1-merge_threshold)
    img_np[prediction_np_q[:, :, 0] > 128, 2]  +=  merge_group[0]*(1-merge_threshold)
    cv2.imwrite(os.path.join(save_mask_path,str(merge_group[0])+'_'+str(merge_group[1])+'_'+str(merge_group[2])+'_'+str(merge_threshold),image_name[:-4] + '.png'), img_np)


import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from config import config
import model
from data_generator import DataGeneraotr
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter1d
from plot import *
from post_proc import *
from post_proc import get_instance_masks
import cv2
from post_proc import get_keypoints

multiscale = [1, 1, 1]
save_path = './demo_result/'  ##/content/drive/My Drive/StrongPose/demo_result

# build the model
batch_size, height, width = 1, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]
tf_img = []
outputs = []
for i in range(len(multiscale)):
    scale = multiscale[i]
    tf_img.append(tf.placeholder(tf.float32, shape=[batch_size, int(scale * height), int(scale * width), 3]))
    outputs.append(model.model(tf_img[i]))
sess = tf.Session()

# load the parameters
global_vars = tf.global_variables()
saver = tf.train.Saver(var_list=global_vars)
checkpoint_path = './model/152/resnet_v2_152.ckpt' ##'./content/drive/My Drive/StrongPose/model/StrongPose/'+'model.ckpt-11'

saver.restore(sess, checkpoint_path)
print("Trained Model Restored!")

# input the demo image
dataset = DataGeneraotr()

scale_outputs = []
for i in range(len(multiscale)):
    scale = multiscale[i]
    scale_img = dataset.get_multi_scale_img(give_id=872, scale=scale)
    if i == 0:
        img = scale_img[:, :, [2, 1, 0]]
        img_img = scale_img[:, :, [2, 1, 0]]
        plt.imsave(save_path + 'input_image.jpg', img)
    imgs_batch = np.zeros((batch_size, int(scale * height), int(scale * width), 3))
    imgs_batch[0] = scale_img

    # make prediction
    one_scale_output = sess.run(outputs[i], feed_dict={tf_img[i]: imgs_batch})
    scale_outputs.append([o[0] for o in one_scale_output])

sample_output = scale_outputs[0]
for i in range(1, len(multiscale)):
    for j in range(len(sample_output)):
        sample_output[j] += scale_outputs[i][j]
for j in range(len(sample_output)):
    sample_output[j] /= len(multiscale)

# visualization
print('Visualization image has been saved into ' + save_path)


def overlay(img, over, alpha):
    out = img.copy()
    if img.max() > 2.:
        out = out / 400.
    out *= 1 - alpha
    if len(over.shape) == 2:
        out += alpha * over[:, :, np.newaxis]
    else:
        out += alpha * over
    return out


#####################################################################################################################################
# Strong Keypoint Heat Map (SKHM)
# Gaussian filtering helps when there are multiple local maxima for the same keypoint.
H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])

#Tune = gaussian_filter(H[:,:,0], sigma=1),
#gaussian_filter(H[:,:,1], sigma=1)
#gaussian_filter(H[:,:,2], sigma=1)
#gaussian_filter(H[:,:,3], sigma=1)
#gaussian_filter(H[:,:,4], sigma=1)
#gaussian_filter(H[:,:,5], sigma=1),
#gaussian_filter(H[:,:,6], sigma=1),
#gaussian_filter(H[:,:,7], sigma=1),
#gaussian_filter(H[:,:,8], sigma=1),
#gaussian_filter(H[:,:,9], sigma=1),
#gaussian_filter(H[:,:,10], sigma=1),
#gaussian_filter(H[:,:,11], sigma=1),
#gaussian_filter(H[:,:,12], sigma=1),
#gaussian_filter(H[:,:,13], sigma=1),
#gaussian_filter(H[:,:,14], sigma=1),
#gaussian_filter(H[:,:,15], sigma=1),
#gaussian_filter(H[:,:,16], sigma=1)

Tune = gaussian_filter(H, sigma=0.4)
for i in range(17):
    H[:, :, i] = gaussian_filter(H[:, :, i], sigma=1)
    # H[:,:,i] = gaussian_filter(H[:,:,i], sigma=0.5)
    # plt.imsave(save_path+'heatmaps.jpg',H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)

    heat_img = plt.imsave(save_path+'SKHM.jpg',H[:,:,0]+H[:,:,1]+H[:,:,2]+H[:,:,3]+H[:,:,4]+H[:,:,5]+H[:,:,6]+H[:,:,7]+H[:,:,8]+H[:,:,9]+H[:,:,10]+H[:,:,11]+H[:,:,12]+H[:,:,13]+H[:,:,14]+H[:,:,15]+H[:,:,16])

######################################################################################################################################

# The heatmaps are computed using the short offsets predicted by the network
# Here are the right shoulder offsets

# points = [14]
# for keypoint_id in points:
#    point= keypoint_id
visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Reye', img=img, every=4, save_path=save_path)

# The connections between keypoints are computed via the mid-range offsets.
# We can visuzalize them as well; for example right shoulder -> right hip
# visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8,save_path=save_path)

# And we can see the reverse connection (Rhip -> Rshjoulder) as well
# visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8,save_path=save_path)
#######################################################################################################################################
# We can use the heatmaps to compute the skeletons
pred_kp = get_keypoints(H)
#print(pred_kp)
pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
pred_skels = [skel for skel in pred_skels if (skel[:, 2] > 0).sum() > 4]
print('Number of detected skeletons: {}'.format(len(pred_skels)))

######################################################################################################################################
# Body Heat Map (BHM)
Rshoulder_map = sample_output[0][:, :, config.KEYPOINTS.index('Rshoulder')]
Lknee = sample_output[0][:, :, config.KEYPOINTS.index('Lknee')]
nose = sample_output[0][:, :, config.KEYPOINTS.index('nose')]
Rear = sample_output[0][:, :, config.KEYPOINTS.index('Rear')]
Lear = sample_output[0][:, :, config.KEYPOINTS.index('Lear')]
Lshoulder = sample_output[0][:, :, config.KEYPOINTS.index('Lshoulder')]
Relbow = sample_output[0][:, :, config.KEYPOINTS.index('Relbow')]
Rwrist = sample_output[0][:, :, config.KEYPOINTS.index('Rwrist')]
Lelbow = sample_output[0][:, :, config.KEYPOINTS.index('Lelbow')]
Lwrist = sample_output[0][:, :, config.KEYPOINTS.index('Lwrist')]
Rhip = sample_output[0][:, :, config.KEYPOINTS.index('Rhip')]
Lhip = sample_output[0][:, :, config.KEYPOINTS.index('Lhip')]
Rknee = sample_output[0][:, :, config.KEYPOINTS.index('Rknee')]
Lankle = sample_output[0][:, :, config.KEYPOINTS.index('Lankle')]
Rankle = sample_output[0][:, :, config.KEYPOINTS.index('Rankle')]
BHM_HOT = Rshoulder_map + Lknee + nose + Lshoulder + Relbow + Rwrist + Lelbow + Lwrist + Rhip + Rknee + Lankle + Lhip + Rankle + Rear + Lear
result = overlay(img, BHM_HOT / 3, alpha=0.4)
cv.putText(result, ('Detected humans: {}'.format(len(pred_skels))), (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5,
           (255, 255, 255), 1)
plt.imsave(save_path + 'BHM.png', result)

##########################################################################################################################################################

BHM_Pose(result, pred_skels, save_path=save_path)

# we can use the predicted skeletons along with the long-range offsets and binary segmentation mask to compute the instance masks.
# plt.imsave(save_path+'segmentation_mask.jpg',apply_mask(img, sample_output[4][:,:,0]>=0.3, color=[0,0,0]))


# plt.imsave(save_path+'Segmentation_Mask.png',apply_mask(img, sample_output[4][:,:,0]>0.5, color=[255,255,235]))


# visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8,save_path=save_path)

# Seg_offset(offsets=sample_output[1], keypoint_id='Rshoulder', img=img, every=8,save_path=save_path)

# shape_mask=get_instance_masks

# print(shape_mask)

instance_masks = get_instance_masks(pred_skels, sample_output[-1][:, :, 0], sample_output[-2])

# plot_instance_masks(instance_masks, img,save_path=save_path)

# print(MaskShape) fail

plot_poses(img_img, pred_skels, save_path=save_path)

Segmentation_Pose(img, pred_skels, instance_masks, save_path=save_path)

# Pose_Segmentation(img, pred_skels, instance_masks, save_path=save_path)



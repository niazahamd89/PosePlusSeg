import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf
from config import config
import model
from data_generator import DataGeneraotr
import numpy as np
from skimage import io
from plot import *
from post_proc import *
import cv2
from post_proc import get_keypoints
from imantics import Polygons, Mask

multiscale = [1,1,1]
save_path = './demo_result/' ##/content/drive/My Drive/StrongPose/demo_result

f = open('./minival1.txt', 'r')   #############'./val_AP.txt', 'r'
s = f.readline()
index = 0 

for s in f:
    print("i is "+str(index))
    print(type(index))
    # build the model
    batch_size = 1
    id_number = int(s.split(",")[0])
    # new = "    IMAGE_SHAPE = (" + s.split(",")[1] + ", " + s.split(",")[2] + ", 3)\n"
    # config_f = open('config.py')
    # config_s = config_f.readlines()
    # # print(config_s[134:136])
    # config_s[135] = new
    # # print(config_s[135])
    # config_f.close()
    # edit_config_f = open('config.py','w')
    # edit_config_f.writelines(config_s)
    # edit_config_f.close()
    # edit_config = config_s.replace("IMAGE_SHAPE = (427, 640, 3)",new)

    height,width = int(s.split(",")[1]),int(s.split(",")[2])

    tf_img = []
    outputs = []
    for i in range(len(multiscale)):
        scale = multiscale[i]
        tf_img.append(tf.compat.v1.placeholder(tf.float32,shape=[batch_size,int(scale*height),int(scale*width),3]))
        outputs.append(model.model(tf_img[i],height,width))
    sess = tf.compat.v1.Session()

    # load the parameters
    global_vars = tf.compat.v1.global_variables()
    saver = tf.compat.v1.train.Saver(var_list = global_vars)
    checkpoint_path = '/content/drive/My Drive/StrongPose/model/StrongPose/model.ckpt-20'
    saver.restore(sess,checkpoint_path)
    print("Trained Model Restored!")

    # input the demo image
    dataset = DataGeneraotr()

    scale_outputs = []
    for i in range(len(multiscale)):
        scale = multiscale[i]
        scale_img = dataset.get_multi_scale_img(give_id=id_number,scale=scale)
        # int(id_number)
        if i==0:
            img = scale_img[:,:,[2,1,0]]
            img_img = scale_img[:,:,[2,1,0]]
            plt.imsave(save_path+'input_image.jpg',img)
        imgs_batch = np.zeros((batch_size,int(scale*height),int(scale*width),3))
        imgs_batch[0] = scale_img

        # make prediction
        one_scale_output = sess.run(outputs[i],feed_dict={tf_img[i]:imgs_batch})
        scale_outputs.append([o[0] for o in one_scale_output])

    sample_output = scale_outputs[0]
    for i in range(1,len(multiscale)):
        for j in range(len(sample_output)):
            sample_output[j]+=scale_outputs[i][j]
    for j in range(len(sample_output)):
        sample_output[j] /=len(multiscale)
    # np.savetxt(['fname', sample_output, "fmt='%.18e'", "delimiter=' '", "newline='\\n'", "header=''", "footer=''", "comments='# '", 'encoding=None']) 
    # np.savetxt('numpy.txt',sample_output[4].reshape((3,-1)),fmt='%d', header=str(array.shape))
    # print(sample_output[4])
    # visualization
    print('Visualization image has been saved into '+save_path)

    def overlay(img, over, alpha=1):
        out = img.copy()
        if img.max() > 2.:
            out = out / 400.
        out *= 1-alpha
        if len(over.shape)==2:
            print(np.newaxis)
            out += alpha*over[:,:,np.newaxis]
        else:
            out += alpha*over
        return out

    #####################################################################################################################################
    # Strong Keypoint Heat Map (SKHM)
    # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
    #H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
    #for i in range(17):
    #    H[:,:,i] = gaussian_filter(H[:,:,i], sigma=1)
    #plt.imsave(save_path+'heatmaps.jpg',H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)

    NonTune = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
    Tune = gaussian_filter(NonTune, sigma=0.1)
    #for i in range(17):
     #   H[:,:,i] = gaussian_filter(H[:,:,i], sigma=1)
    #H[:,:,i] = gaussian_filter(H[:,:,i], sigma=0.5)
#plt.imsave(save_path+'heatmaps.jpg',H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)
 

        #################heat_img = plt.imsave(save_path+'SKHM.jpg',H[:,:,0]+H[:,:,1]+H[:,:,2]+H[:,:,3]+H[:,:,4]+H[:,:,5]+H[:,:,6]+H[:,:,7]+H[:,:,8]+H[:,:,9]+H[:,:,10]+H[:,:,11]+H[:,:,12]+H[:,:,13]+H[:,:,14]+H[:,:,15]+H[:,:,16])

    ######################################################################################################################################

    # The heatmaps are computed using the short offsets predicted by the network
    # Here are the right shoulder offsets
    #visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8,save_path=save_path)

    # The connections between keypoints are computed via the mid-range offsets.
    # We can visuzalize them as well; for example right shoulder -> right hip
    #visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8,save_path=save_path)

    # And we can see the reverse connection (Rhip -> Rshjoulder) as well
    # visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8,save_path=save_path)
    #######################################################################################################################################
    # We can use the heatmaps to compute the skeletons
    pred_kp = get_keypoints(Tune)
    #result = open('result_niaz.txt','a')
    #result.write(str(pred_kp))
    #result.write("\n")
    #result.close()

    # print(pred_kp)
    pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2],img_id=id_number)
    pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
    print ('Number of detected skeletons: {}'.format(len(pred_skels)))

    ######################################################################################################################################
    #Body Heat Map (BHM)
    #Rshoulder_map = sample_output[0][:,:,config.KEYPOINTS.index('Rshoulder')]
    #Lknee = sample_output[0][:,:,config.KEYPOINTS.index('Lknee')]
    #nose = sample_output[0][:,:,config.KEYPOINTS.index('nose')]
    #Rear = sample_output[0][:,:,config.KEYPOINTS.index('Rear')]
    #Lear = sample_output[0][:,:,config.KEYPOINTS.index('Lear')]
    #Lshoulder = sample_output[0][:,:,config.KEYPOINTS.index('Lshoulder')]
    #Relbow = sample_output[0][:,:,config.KEYPOINTS.index('Relbow')]
    #Rwrist = sample_output[0][:,:,config.KEYPOINTS.index('Rwrist')]
    #Lelbow = sample_output[0][:,:,config.KEYPOINTS.index('Lelbow')]
    #Lwrist = sample_output[0][:,:,config.KEYPOINTS.index('Lwrist')]
    #Rhip = sample_output[0][:,:,config.KEYPOINTS.index('Rhip')]
    #Lhip = sample_output[0][:,:,config.KEYPOINTS.index('Lhip')]
    #Rknee = sample_output[0][:,:,config.KEYPOINTS.index('Rknee')]
    #Lankle = sample_output[0][:,:,config.KEYPOINTS.index('Lankle')]
    #Rankle = sample_output[0][:,:,config.KEYPOINTS.index('Rankle')]
    #result = Rshoulder_map + Lknee + nose +Lshoulder+ Relbow + Rwrist + Lelbow + Lwrist + Rhip + Rknee + Lankle + Lhip + Rankle + Rear + Lear
    #result = overlay(img, result/3, alpha=0.4)
    # cv.putText(result,('Detected Humans: {}'.format(len(pred_skels))),(5,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    # plt.imsave(save_path+'BHM.jpg',result)

    ##########################################################################################################################################################


    # we can use the predicted skeletons along with the long-range offsets and binary segmentation mask to compute the instance masks.
    # plt.imsave(save_path+'segmentation_mask.jpg',apply_mask(img, sample_output[4][:,:,0]>0.9, color=[255,255,200]))
    # seg_prob = sample_output[4][:,:,0]>0.9
    # polygons = Mask(seg_prob).polygons()
    # print(polygons.points)
    # print(polygons.segmentation)
    

    #visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8,save_path=save_path)

    #instance_masks = get_instance_masks(pred_skels, sample_output[-1][:,:,0], sample_output[-2])

    #plot_instance_masks(instance_masks, img,save_path=save_path)

    #plot_poses(img_img, pred_skels,save_path=save_path)

    #Segmentation_Pose(img, pred_skels, instance_masks, id = id_number,save_path=save_path)


    index = index + 1
    
    # #Pose_Segmentation(img, pred_skels, instance_masks, save_path=save_path)
    if index is 100:
        break

f.close()

from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv
import numpy as np
import math
from config import config
from post_proc import get_keypoints
from PIL import Image
from imantics import Polygons, Mask
import sys
np.set_printoptions(threshold=sys.maxsize)

import base64
from pycocotools import _mask as coco_mask
from pycocotools import mask
import typing as t
import zlib
from sklearn.preprocessing import binarize



#############################################################################################
def visualize_short_offsets(offsets, keypoint_id, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None, every=1,save_path='./'):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            Rknee = config.KEYPOINTS.index('Rknee')
            Lelbow = config.KEYPOINTS.index('Relbow')
    keypoint_id = Rknee+Lelbow
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==keypoint_id]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*keypoint_id:2*keypoint_id+2]
    masks = np.zeros(offsets.shape[:2]+(len(centers),), dtype='bool')
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists = np.sqrt(np.square(idx-c).sum(axis=-1))
        dists_x = np.abs(idx[:,:,0] - c[0])
        dists_y = np.abs(idx[:,:,1] - c[1])
        
        masks[:,:,j] = (dists<=radius)
        if every > 1:
            d_mask = np.logical_and(np.mod(dists_x.astype('int32'), every)==0, np.mod(dists_y.astype('int32'), every)==0)
            masks[:,:,j] = np.logical_and(masks[:,:,j], d_mask)
    mask = masks.sum(axis=-1) > 0

    
    
#     for j, c in enumerate(centers):
#         dists[:,:,j] = np.sqrt(np.square(idx-c).sum(axis=-1))
#     dists = dists.min(axis=-1)
#     mask = dists <= radius
    I, J = np.nonzero(mask)

    plt.figure()
    if img is not None:
        plt.imshow(img)

    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200    
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='b', angles='xy', scale_units='xy', scale=1.5)
    # plt.savefig('./demo_result/short_offsets.jpg',bbox_inches = 'tight')
########################################################################################################

def visualize_mid_offsets(offsets, from_kp, to_kp, centers=None, heatmaps=None, radius=config.KP_RADIUS, img=None, every=1,save_path='./'):
    if centers is None and heatmaps is None:
        raise ValueError('either keypoint locations or heatmaps must be provided')
    

    if isinstance(from_kp, str):
        if not from_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(from_kp))
        else:
            from_kp = config.KEYPOINTS.index(from_kp)
    if isinstance(to_kp, str):
        if not to_kp in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(to_kp))
        else:
            to_kp = config.KEYPOINTS.index(to_kp)

    edge_list = config.EDGES + [edge[::-1] for edge in config.EDGES]
    edge_id = edge_list.index((from_kp, to_kp))
    
    if centers is None:
        kp = get_keypoints(heatmaps)
        kp = [k for k in kp if k['id']==from_kp]
        centers = [k['xy'].tolist() for k in kp]

    kp_offsets = offsets[:,:,2*edge_id:2*edge_id+2]
    # dists = np.zeros(offsets.shape[:2]+(len(centers),))
    masks = np.zeros(offsets.shape[:2]+(len(centers),), dtype='bool')
    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    for j, c in enumerate(centers):
        dists = np.sqrt(np.square(idx-c).sum(axis=-1))
        dists_x = np.abs(idx[:,:,0] - c[0])
        dists_y = np.abs(idx[:,:,1] - c[1])
        masks[:,:,j] = (dists<=radius)
        if every > 1:
            d_mask = np.logical_and(np.mod(dists_x.astype('int32'), every)==0, np.mod(dists_y.astype('int32'), every)==0)
            masks[:,:,j] = np.logical_and(masks[:,:,j], d_mask)

    mask = masks.sum(axis=-1) > 0
    # dists = dists.min(axis=-1)
    # mask = dists <= radius
    I, J = np.nonzero(mask)

    if img is not None:
        plt.imshow(img)
        
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    # plt.savefig(save_path+'middle_offsets.jpg',bbox_inches = 'tight')
##################################################################################################################

def visualize_long_offsets(offsets, keypoint_id, seg_mask, img=None, every=1,save_path='./'):
    if isinstance(keypoint_id, str):
        if not keypoint_id in config.KEYPOINTS:
            raise ValueError('{} not a valid keypoint name'.format(keypoint_id))
        else:
            keypoint_id = config.KEYPOINTS.index(keypoint_id)

    idx = np.rollaxis(np.indices(offsets.shape[1::-1]), 0, 3).transpose((1,0,2))
    kp_offsets = offsets[:,:,2*keypoint_id:2*keypoint_id+2]
    mask = seg_mask[:,:,0]>0.5
    mask = np.logical_and(mask, np.mod(idx[:,:,0], every)==0)
    mask = np.logical_and(mask, np.mod(idx[:,:,1], every)==0)
    I, J = np.nonzero(mask)
    
    if img is not None:
        plt.imshow(img)
    
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 200 
    plt.quiver(J, I, kp_offsets[I,J,0], kp_offsets[I,J,1], color='r', angles='xy', scale_units='xy', scale=1)
    # plt.savefig(save_path+'long_offsets.jpg',bbox_inches = 'tight')

######################################################################################################
    
def plot_poses(img, skeletons, save_path=None):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    #plt.figure()
    
    #img = img.astype('uint8')
    canvas = img.copy()

    for i in range(17):
        rgba = np.array(cmap(1 - i/17. - 1./34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)    

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()

    stickwidth = 1

    for i in range(config.NUM_EDGES):
        for j in range(len(skeletons)):
            edge = config.EDGES[i]
            if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            #print(X,Y)
            mX = np.mean(X)
            mY = np.mean(Y)
#            print(mX,mY)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

            cv.putText(canvas,('Detected Skeletons: {}'.format(len(skeletons))),(5,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)    
   
            ####################plt.imsave(save_path+'pose.jpg',canvas[:,:,:])
            fig = matplotlib.pyplot.gcf()

#########################################################################################################

def apply_mask(image, mask, color, alpha=0.5):
#    image = image.copy()
    # print(mask)
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def plot_instance_masks(masks, img, save_path='./'):
    canvas = img.copy()
    i = 0
    for mask in masks:
        color = [np.random.uniform() for _ in range(3)]
        canvas = apply_mask(img, mask, color, alpha=0.50)

        i = i + 1

    ######################plt.imsave(save_path+'instances_masks.jpg',canvas)

    # print(polygons.segmentation)


def plot_mask_pose(image, mask, color, alpha):
    image = image.copy()
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
    
#########################################################################################################





def encode_binary_mask(mask: np.ndarray) -> t.Text:
    """Converts a binary mask into OID challenge encoding ascii text."""

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError("encode_binary_mask expects a binary mask, received dtype == %s" % mask.dtype)

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError("encode_binary_mask expects a 2d mask, received shape == %s" % mask.shape)

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def segmentationToCocoMask(labelMap, labelId):
    '''
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param labelId: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    labelMask = labelMap == labelId
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs




def Segmentation_Pose(img, skeletons, masks, id ,save_path=None):

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()
    
    #img = img.astype('uint8')
    canvas = img.copy()

    for i in range(17):
        rgba = np.array(cmap(1 - i/17. - 1./34))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            cv.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 2, colors[i], thickness=-1)    

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    plt.imshow(to_plot[:,:,[2,1,0]])
    fig = matplotlib.pyplot.gcf()
    print(fig)
    stickwidth = 1
    id_edit = str(id)
    k = 0 
    for i in range(config.NUM_EDGES):
        for j in range(len(skeletons)):
            edge = config.EDGES[i]
            if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
       
            mX = np.mean(X)
            mY = np.mean(Y)
#            print(mX,mY)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
  

            for mask in masks:
                color = [np.random.uniform() for _ in range(3)]
                canvass = plot_mask_pose(canvas, mask, color, alpha=0.30)
            
                cv.putText(canvas,('Detected Skeletons & Masks: {}'.format(len(skeletons))),(5,30),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1)    

                # plt.imsave(save_path+'Seg_Pose_'+ id_edit +'.jpg',canvass[:,:,:])
                fig = matplotlib.pyplot.gcf()

                # plt.imsave(save_path+'fuck_' + str(i) + '.jpg',mask)
                
                # mask_bin = np.array(mask, dtype=np.float32).tobytes()
                mask_bool = np.array(mask, dtype=bool)
                # print(mask_bin)
                encoded_mask = segmentationToCocoMask(mask_bool,1)

                # encoded_mask = encode_binary_mask(mask_bool)
                
                # polygons = Mask(mask).polygons()
                # k = k + 1
                # polygon_array = np.array(polygons.points,dtype = object)
                
                # polygon_array = polygon_array.tolist()
                
                # # print(polygon_array)
                # # print(type(polygon_array))
                if k == 0:
                    result = open('segmentation_niaz.txt','a')
                    result.write(str(id_edit)+"|"+str(encoded_mask))
                    result.write("\n")
                    result.close()

            k = k + 1

    # print(encoded_mask)


####################################################################################################
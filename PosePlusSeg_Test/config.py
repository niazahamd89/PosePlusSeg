
class config:

    #########
    # POSE CONFIGS:
    #########

    # Number of keypoints
    NUM_KP = 17

    # List of keypoint names person lab order
    KEYPOINTS = [
        "nose",         # 0
        # "neck",       
        "Rshoulder",    # 1
        "Relbow",       # 2
        "Rwrist",       # 3
        "Lshoulder",    # 4
        "Lelbow",       # 5
        "Lwrist",       # 6
        "Rhip",         # 7
        "Rknee",        # 8
        "Rankle",       # 9
        "Lhip",         # 10
        "Lknee",        # 11
        "Lankle",       # 12
        "Reye",         # 13
        "Leye",         # 14
        "Rear",         # 15
        "Lear",         # 16
    ]
   
    

    # Indices of right and left keypoints (for flipping in augmentation)
    RIGHT_KP = [1, 2, 3,  7,  8,  9, 13, 15]
    LEFT_KP =  [4, 5, 6, 10, 11, 12, 14, 16]

    # List of edges as tuples of indices into the KEYPOINTS array
    # (Each edge will be used twice in the mid-range offsets; once in each direction)
    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]

    NUM_EDGES = len(EDGES)
    
     ############################## List of keypoint names coco order#########################
    #KEYPOINTS = [
     #   "nose",          # 0
        # "neck",       
     #   "Leye",          # 1
     #   "Reye",          # 2
     #   "Lear",          # 3
      #  "Rear",          # 4
      #  "Lshoulder",     # 5
      #  "Rshoulder",     # 6
      #  "Lelbow",        # 7
      #  "Relbow",        # 8
      #  "Lwrist",        # 9
      #  "Rwrist",        # 10
      #  "Lhip",          # 11
      #  "Rhip",          # 12
      #  "Lknee",         # 13
      #  "Rknee",         # 14
      #  "Lankle",        # 15
      #  "Rankle",        # 16
   # ]
    
    
    
    # Indices of right and left keypoints (for flipping in augmentation) for COCO
  #  RIGHT_KP = [6, 8, 10,  12,  14,  16, 2, 4]
   # LEFT_KP =  [5, 7,  9,  11,  13,  15, 1, 3]
    
    #List of edges fro coco
   # EDGES = [
   #     (0, 1),
   #     (0, 2),
   #     (0, 5),
   #     (0, 6),
   #     (1, 3),
   #     (2, 4),
   #     (5, 11),
   #     (6, 12),
   #     (11, 13),
   #     (12, 14),
   #     (13, 15),
   #     (14, 16),
   #     (5, 7),
   #     (6, 8),
   #     (7, 9),
   #     (8, 10)
   # ]
    
  #  NUM_EDGES = len(EDGES)
    ############################## List of keypoint names coco order#########################

    #########
    # PRE- and POST-PROCESSING CONFIGS:
    #########

    # Radius of the discs around the keypoints. Used for computing the ground truth
    # and computing the losses. (Recommended to be a multiple of the output stride.)
    KP_RADIUS = 32

    # The threshold for extracting keypoints from hough maps.
    PEAK_THRESH = 0.001           

    # Pixel distance threshold for whether to begin a new skeleton instance
    # (If another skeleton already has this keypoint within the threshold, it is discarded.)
    NMS_THRESH = 32

    # The metric threshold for assigning a pixel to a given instance mask 
    INSTANCE_SEG_THRESH = 0.25

    #########
    # TRAINING CONFIGS:
    #########

    # Input shape for training images (By convention s*n+1 for some integer n and s=output_stride)
    IMAGE_SHAPE = (230, 352, 3)

    # Output stride of the base network (resnet101 or resnet152 in the paper)
    # [Any convolutional stride in the original network which would reduce the 
    # output stride further is replaced with a corresponding dilation rate.]
    OUTPUT_STRIDE = 8 #16 for training 8 for evaluation. demo_result

    # Weights for the losses applied to the keypoint maps ('heatmap'), the binary segmentation map ('seg'),
    # and the short-, mid-, and long-range offsets.
    '''
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'seg': 2,
        'short': 1,
        'mid': 0.25,
        'long': 0.125
    }
    '''
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'seg': 2,
        'short': 1,
        'mid': 0.4,
        'long': 0.1
    }
    # Batch_size
    BATCH_SIZE = 2

    # Learning Rate
    LEARNING_RATE = 0.5e-3

    #def Lr_schedule(epoch):
        #lr = 1e-3
        #if epoch>180:
        #    lr*= 0.5e-3
        #elif epoch > 160:
         #   lr*= 1e-3
        #elif epoch > 120:
         #   lr *= 1e-2
        #elif epoch >80:
        #    lr *= 1e-1
        #print('Learning rate: ', lr)
        #return lr



    # Whether to keep the batchnorm weights frozen.
    BATCH_NORM_FROZEN = True

    # Number of GPUs to distribute across
    NUM_GPUS = 1

    # The total batch size will be (NUM_GPUS * BATCH_SIZE_PER_GPU)
    BATCH_SIZE_PER_GPU = 1

    # Whether to use Polyak weight averaging as mentioned in the paper
    POLYAK = True



    # Optional model weights filepath to use as initialization for the weights
    LOAD_MODEL_PATH = None

    # Where to save the model.
    SAVE_MODEL_PATH = './model/StrongPose/'

    # Where to save the pretrained model
    PRETRAINED_MODEL_PATH = './model/101/resnet_v2_101.ckpt'
    #PRETRAINED_MODEL_PATH = './model/StrongPose/model.ckpt-11'


    # Epochs
    NUM_EPOCHS = 500
    
    # Epoch_sizes
    NUM_EPOCHS_SIZE = 10000
    
    # Where to save the coco2017 dataset
    #ANNO_FILE = 'COCO2017/annotations/person_keypoints_train2017.json'
    #MG_DIR = 'COCO2017/train2017/'
    
    #ANNO_FILE = './COCO2017/annotations/image_info_test2017.json'
    #IMG_DIR = './COCO2017/test2017/'



    # Where to save the coco2017 dataset
    ANNO_FILE = 'COCO2017/annotations/person_keypoints_val2017'         #val_ochuman_coco.json // person_keypoints_val2017
    MG_DIR = 'COCO2017/val2017/'                                        #OCHuman   //   val2017
    
    
    # log dir
    LOG_DIR = './log'

class TransformationParams:

    target_dist = 0.8
    scale_prob = 1.
    scale_min = 0.8
    scale_max = 2.0
    max_rotate_degree = 30.
    center_perterb_max = 20.0
    flip_prob = 0.5

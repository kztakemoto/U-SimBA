#################################################################################
# argument parser
#################################################################################
#    --dataset: str, 'melanoma', 'oct', or 'chestx', dataset type
#    --model_path: str, path to model weight '*/*.h5'
#    --model_type: 'InceptionV3', 'VGG16', 'ResNet50'
#    --norm_type: str, '2' or 'inf', norm type of UAPs
#    --norm_rate: float, noise strength (zeta)
#    --epsilon: float, attack strength (step size)
#    --max_iter: int, maximum number of iterations for computing UAP (i_max)
#    --freqdim: int, frequency dimension for Q_DCT
#    --nb_samples: int, input dataset size
#    --targeted: int, target class (negative value indicates non-targeted attacks)
#    --save_path: str, path to output files 
#    --gpu: str, for os.environ["CUDA_VISIBLE_DEVICES"]
#################################################################################

import warnings
warnings.filterwarnings('ignore')
import os, sys, gc, pdb, argparse
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras import utils

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Lambda, Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD

from art.classifiers import KerasClassifier
from art.attacks.evasion import Universal_SimBA
from art.utils import random_sphere
from art.utils import projection

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed=200):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

seed=123

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# check the starting time for computing total processing time
import time
start_time = time.time()

# set labels
label2nb_dict = {
    'chestx':
        {'NORMAL': 0, 'PNEUMONIA': 1},
    'oct':
        {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3},
    'melanoma':
        {'MEL': 0, 'NV': 1, 'BCC': 2,
         'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
}

batch_size= 256

### UAP class ###
# classifier: classifier
# X_train: ndarray, input images
# y_train: ndarray, the labels of the input images
# X_test: ndarray, validation images
# y_test: ndarray, the labels of the validation images
# X_original_train: ndarray, training images
# y_original_train: ndarray, the labels of the training images
# norm_type: 2 or np.inf, norm type of UAPs
# norm_size: float, noise size (xi)
# epsilon: float, attack strength (step size)
# freqdim: int, frequency dimension for Q_DCT
# max_iter: int, maximum number of iterations for computing UAP.
# targeted: int, target class (negative value indicates non-targeted attacks)
# save_path: str, path to output files 
class my_UAP:
    def __init__(
                self,
                classifier,
                X_train, y_train,
                X_test, y_test,
                X_original_train, y_original_train,
                norm_type,
                norm_size,
                epsilon,
                freqdim,
                max_iter,
                targeted,
                save_path,
                ):

        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_original_train = X_original_train
        self.y_original_train = y_original_train
        self.norm_type = norm_type
        self.norm_size = norm_size
        self.epsilon = epsilon
        self.freqdim = freqdim
        self.max_iter = max_iter
        self.targeted = targeted
        self.save_path = save_path

    ### compute the attack success rate
    # images: ndarray, target image set
    # noise: ndarray, UAP
    def my_calc_fooling_ratio(self, images=0, noise=0):
        adv_images = images + noise
        if self.targeted < 0:
            preds = np.argmax(self.classifier.predict(images, batch_size=batch_size), axis=1)
            preds_adv = np.argmax(self.classifier.predict(adv_images, batch_size=batch_size), axis=1)
            fooling_ratio = np.sum(preds_adv != preds) / images.shape[0]
            return fooling_ratio
        else:
            preds_adv = np.argmax(self.classifier.predict(adv_images, batch_size=batch_size), axis=1)
            fooling_ratio_targeted = np.sum(preds_adv == self.targeted) / adv_images.shape[0]
            return fooling_ratio_targeted

    ### generate the labels (in one-hot vector representation) for targeted attacks
    # length: int, number of target images
    def my_target_labels(self, length=0):
        classes = self.y_train.shape[1]
        return utils.to_categorical([self.targeted] * length, classes)

    ### generate UAP
    def my_gen_UAP(self):
        imshape = self.X_train[0].shape

        if self.targeted >= 0:
            print(" *** targeted attack *** \n")
            adv_crafter = Universal_SimBA(
                self.classifier,
                attack='dct',
                epsilon=self.epsilon,
                freq_dim=self.freqdim,
                max_iter=self.max_iter,
                eps=self.norm_size,
                norm=self.norm_type,
                targeted=True,
                batch_size=batch_size
            )
        else:
            print(" *** non-targeted attack *** \n")
            adv_crafter = Universal_SimBA(
                self.classifier,
                attack='dct',
                epsilon=self.epsilon,
                freq_dim=self.freqdim,
                max_iter=self.max_iter,
                eps=self.norm_size,
                norm=self.norm_type,
                targeted=False,
                batch_size=batch_size
            )

        # initialization
        LOG = []
        X_materials_cnt = 0
        X_materials = self.X_train

        # craft UAP
        if self.targeted >= 0:
            # generate the one-hot vector of the target label
            Y_materials_tar = self.my_target_labels(length=len(X_materials))
            _ = adv_crafter.generate(X_materials,  y=Y_materials_tar)
        else:
            _ = adv_crafter.generate(X_materials)

        # handling for no noise
        if type(adv_crafter.noise[0,:]) == int:
            noise = np.zeros(imshape)
        else:
            noise = np.copy(adv_crafter.noise)
            noise = np.reshape(noise, imshape)

        # generate random UAP whose size equals to the size of the UAP
        noise_size = float(np.linalg.norm(noise.reshape(-1), ord=self.norm_type))
        noise_random = random_sphere(
            nb_points=1,
            nb_dims=np.prod(X_materials[0].shape),
            radius=noise_size,
            norm=self.norm_type
        ).reshape(imshape)

        # compute attack success rate of UAP
        # for input data
        fr_train = self.my_calc_fooling_ratio(images=self.X_train, noise=noise)
        # for validation data
        fr_test = self.my_calc_fooling_ratio(images=self.X_test, noise=noise)
        # for training data
        fr_m = self.my_calc_fooling_ratio(images=self.X_original_train, noise=noise)

        # compute attack success rate of random UAP
        # for input data
        fr_train_r = self.my_calc_fooling_ratio(images=self.X_train, noise=noise_random)
        # for validation data
        fr_test_r = self.my_calc_fooling_ratio(images=self.X_test, noise=noise_random)
        # for training data
        fr_m_r = self.my_calc_fooling_ratio(images=self.X_original_train, noise=noise_random)

        # compute UAP size
        norm_2 = np.linalg.norm(noise)
        norm_inf = abs(noise).max()

        LOG.append([X_materials_cnt, norm_2, norm_inf, fr_train, fr_test, fr_m, fr_train_r, fr_test_r, fr_m_r])
        print("LOG: {} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(X_materials_cnt, norm_2, norm_inf, fr_train, fr_test, fr_m, fr_train_r, fr_test_r, fr_m_r))

        np.save(self.save_path+'/noise', noise)
        np.save(self.save_path+'/LOG', np.array(LOG))
        return noise, np.array(LOG)


### cofiguration of classifier
# model_type: 'InceptionV3', 'VGG16', 'ResNet50'
# model_path: str, path to model weight
# output_class: int, number of classes
# mono: int, monochrome images if mono = 1, RGB images otherwise
# silence: int, prevent to output model summary if silence = 1, not otherwise
class my_DNN:
    def __init__(
                self,
                model_type,
                model_path,
                output_class,
                mono,
                silence
                ):
        self.model_type = model_type
        self.model_path = model_path
        self.output_class = output_class
        self.mono = mono
        self.silence = silence

    def my_classifier(self):
        if self.mono==1:
            input_shape = (299, 299, 3)
            if self.model_type == 'inceptionv3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)
            elif self.model_type == 'vgg16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights=None, input_shape=input_shape, include_top=False)
            elif self.model_type == "resnet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
            else:
                print(" --- ERROR : UNKNOWN MODEL TYPE --- ")
            base_model.layers.pop(0)
            newInput = Input(batch_shape=(None, 299,299,1))
            x = Lambda(lambda image: tf.image.grayscale_to_rgb(image))(newInput)
            tmp_out = base_model(x)
            tmpModel = Model(newInput, tmp_out)
            x = tmpModel.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(tmpModel.input, predictions)

        else:
            input_shape = (299, 299, 3)
            if self.model_type == 'inceptionv3':
                print(" MODEL: InceptionV3")
                base_model = InceptionV3(weights='imagenet', input_shape=input_shape, include_top=False)
            elif self.model_type == 'vgg16':
                print(" MODEL: VGG16")
                base_model = VGG16(weights='imagenet', input_shape=input_shape, include_top=False)
            elif self.model_type == "resnet50":
                print(" MODEL: ResNet50")
                base_model = ResNet50(weights='imagenet', input_shape=input_shape, include_top=False)
            else:
                print(" --- ERROR: UNKNOWN MODEL TYPE --- ")
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(self.output_class, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True

        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(self.model_path,)
        if self.silence != 1:
            model.summary()
        classifier = KerasClassifier(model=model)

        print("Finish Load Model")
        return classifier


### Main ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--norm_type', type=str)
    parser.add_argument('--norm_rate', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--freqdim', type=int)
    parser.add_argument('--nb_samples', type=int)
    parser.add_argument('--targeted', type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")

    os.makedirs(args.save_path, exist_ok=False)
    handler2 = logging.FileHandler(filename=f"{args.save_path}/log.txt")
    handler2.setLevel(logging.INFO)
    handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler2)

    if args.norm_type == '2':
        norm_type = 2
    elif args.norm_type == 'inf':
        norm_type = np.inf
    norm_rate = args.norm_rate

    # load data
    X_train = np.load(f"./data/{args.dataset}/X_train.npy")
    y_train = np.load(f"./data/{args.dataset}/y_train.npy")
    X_test_1 = np.load(f"./data/{args.dataset}/X_test_1_{args.nb_samples}.npy")
    y_test_1 = np.load(f"./data/{args.dataset}/y_test_1_{args.nb_samples}.npy")
    X_test_2 = np.load(f"./data/{args.dataset}/X_test_2_{args.nb_samples}.npy")
    y_test_2 = np.load(f"./data/{args.dataset}/y_test_2_{args.nb_samples}.npy")

    # check color type (mono or RGB)
    if X_train.shape[-1] != 3:
        mono = 1
    else:
        mono = 0

    # compute the actual norm size from the ratio `norm_rate` of the Lp of the UAP to the average Lp norm of an image in the dataset (training images)
    if norm_type == np.inf:
        norm_mean = 0
        for img in X_test_1:
            norm_mean += abs(img).max()
        norm_mean = norm_mean/X_test_1.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        logger.info("\n ------------------------------------")
        logger.info(" Linf norm: {:.2f} ".format(norm_size))
    else:
        norm_mean = 0
        for img in X_test_1:
            norm_mean += np.linalg.norm(img)
        norm_mean = norm_mean/X_test_1.shape[0]
        norm_size = float(norm_rate*norm_mean/128.0)
        logger.info(" L2 norm: {:.2f} ".format(norm_size))

    # normalization
    X_train -= 128.0
    X_train /= 128.0
    X_test_1 -= 128.0
    X_test_1 /= 128.0
    X_test_2 -= 128.0
    X_test_2 /= 128.0

    logger.info(f"Train Size: {y_train.shape}")
    logger.info(f"Test Size: {y_test_1.shape}")
    logger.info(f"Eval  Size: {y_test_2.shape}")

    dnn = my_DNN(
        model_type=args.model_type,
        model_path=args.model_path,
        output_class=y_train.shape[1],
        mono=mono,
        silence=1
    )
    classifier = dnn.my_classifier()

    # compute the accuracies for clean images
    preds_train = np.argmax(classifier.predict(X_train, batch_size=batch_size), axis=1)
    acc = np.sum(preds_train == np.argmax(y_train, axis=1)) / y_train.shape[0]
    logger.info(" Accuracy [train]: {:.2f}".format(acc))
    preds_test1 = np.argmax(classifier.predict(X_test_1, batch_size=batch_size), axis=1)
    acc = np.sum(preds_test1 == np.argmax(y_test_1, axis=1)) / y_test_1.shape[0]
    logger.info(" Accuracy [test 1]: {:.2f}".format(acc))
    preds_test2 = np.argmax(classifier.predict(X_test_2, batch_size=batch_size), axis=1)
    acc = np.sum(preds_test2 == np.argmax(y_test_2, axis=1)) / y_test_2.shape[0]
    logger.info(" Accuracy [test 2]: {:.2f}".format(acc))
    logger.info(" ------------------------------------\n")

    # generate UAP
    uap = my_UAP(
                classifier=classifier,
                X_train=X_test_1, y_train=y_test_1,
                X_test=X_test_2, y_test=y_test_2,
                X_original_train=X_train, y_original_train=y_train,
                norm_type=norm_type,
                norm_size=norm_size,
                epsilon=args.epsilon,
                freqdim=args.freqdim,
                max_iter=args.max_iter,
                targeted=args.targeted,
                save_path=args.save_path,
                )
    noise, LOG = uap.my_gen_UAP()

    # output processing time
    processing_time = time.time() - start_time
    logger.info("\n\t ------------------------------------")
    logger.info("\t   total processing time : {:.2f} h.".format(processing_time/3600))
    logger.info("\t ------------------------------------\n")

    # save figures
    save_f_img = f'{args.save_path}/sample.png'
    make_adv_img(
        clean_img=X_test_1[0],
        noise=noise,
        adv_img=X_test_1[0] + noise,
        save_file_name=save_f_img,
        nlz="11"
    )
    
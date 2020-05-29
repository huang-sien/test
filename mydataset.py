import tensorflow as tf
import sys
nets_path = 'slim'
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
    #print("!!!!!!!!!!!!!!!!!!!!!!")
    #print(sys.path.insert(0,nets_path))
else:
    print('ok slim')

from slim.nets.nasnet import nasnet

slim = tf.contrib.slim
image_size = nasnet.build_nasnet_mobile.default_image_size #331

from preprocessing import preprocessing_factory

import os

def list_images(directory):
    labels = os.listdir(directory)
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            if 'jpg' in f.lower() or 'png' in f.lower():
                files_and_labels.append((os.path.join(directory,label,f),label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    
    for i, label in enumerate (sorted(unique_labels)):
        label_to_int[label] = i

    print(label,label_to_int[label])

    labels = [label_to_int[l] for l in labels]
    print(labels[:6],labels[-6:])
    return filenames, labels

num_workers=1

image_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile',is_training=True)
image_eval_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile',is_training=False)


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image, label


def training_preprocess(image, label):
    image = image_preprocessing_fn(image, image_size, image_size)
    return image, label


def val_preprocess(image, label):
    image = image_eval_preprocessing_fn(image, image_size, image_size)
    return image, label


def creat_batched_dataset(filenames, labels, batch_size, isTrain=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function, num_parallel_calls=num_workers)

    if isTrain==True:
        dataset = dataset.shuffle(buffer_size=len(filenames))
        dataset = dataset.map(training_preprocess, num_parallel_calls=num_workers)

    else:
        dataset = dataset.map(val_preprocess, num_parallel_calls=num_workers)

    return dataset.batch(batch_size)

def creat_dataset_fromdir(directory, batch_size, isTrain=True):
    filenames, labels = list_images(directory)
    num_classes = len(set(labels))
    dataset = creat_batched_dataset(filenames, labels, batch_size, isTrain)
    return dataset, num_classes
























    

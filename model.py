import sys
nets_path = 'slim'
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('ok slim')

import tensorflow as tf

from slim.nets.nasnet import nasnet
slim = tf.contrib.slim

import os
#import 
mydataset = __import__("mydataset")
creat_dataset_fromdir = mydataset.creat_dataset_fromdir

class MyNASNetModel(object):
    def __init__(self, model_path='/data/home/qaz147652/fine_tune/mango_nasnet/ckpt/model.ckpt'):
        self.model_path = model_path

    def MyNASNet(self,images,is_training):
        arg_scope = nasnet.nasnet_mobile_arg_scope()
        with slim.arg_scope(arg_scope):
            logits, end_points = nasnet.build_nasnet_mobile(images, num_classes = self.num_classes+1, is_training=is_training)
    
        global_step = tf.train.get_or_create_global_step()

        return logits, end_points, global_step


    def FineTuneNASNet(self, is_training):
        model_path = self.model_path

        exclude = ['final_layer','aux_7']

        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        if is_training==True:
            init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore)
        else:
            init_fn = None

        tuning_variables = []
        for v in exclude:
            tuning_variables += slim.get_variables(v)

        return init_fn, tuning_variables

    def build_acc_base(self, labels):
        self.prediction = tf.cast(tf.argmax(self.logits, 1),tf.int32)
        self.correct_prediction = tf.equal(self.prediction, labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,dtype=tf.float32))
        self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=labels, k=5),dtype=tf.float32))

    def load_cpk(self, global_step, sess, begin = 0, saver=None, save_path = None):
        if begin==0:
            save_path = './train_nasnet'
            if not os.path.exists(save_path):
                print('no model path')
            saver = tf.train.Saver(max_to_keep=1)
            return saver, save_path

        else:
            kpt = tf.train.latest_checkpoint(save_path)
            print('load model')
            startepo=0
            if kpt!=None:
                saver.restore(sess,kpt)
                ind = kpt.find("-")
                startepo = int(kpt[ind+1:])
                print('global_step=',global_step.eval(), startepo)
            return startepo

    def build_model_train(self,images,labels,learning_rate1,learning_rate2,is_training):
        self.logits,self.end_points,self.global_step = self.MyNASNet(images,is_training=is_training)
        self.step_init = self.global_step.initializer
        #print(self.global_step)

        self.init_fn,self.tuning_variables = self.FineTuneNASNet(is_training=is_training)

        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        loss = tf.losses.get_total_loss()

        learning_rate1 = tf.train.exponential_decay(learning_rate = learning_rate1, global_step = self.global_step,decay_steps=100, decay_rate=0.5)
        learning_rate2 = tf.train.exponential_decay(learning_rate = learning_rate2, global_step = self.global_step,decay_steps=100, decay_rate=0.2)
        
        last_optimizer = tf.train.AdamOptimizer(learning_rate1)
        full_optimizer = tf.train.AdamOptimizer(learning_rate2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.last_train_op = last_optimizer.minimize(loss,self.global_step,var_list = self.tuning_variables)
            self.full_train_op = full_optimizer.minimize(loss,self.global_step)

    
        self.build_acc_base(labels)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('accuracy_top_5', self.accuracy_top_5)

        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter('./log_dir/train')
        self.eval_writer = tf.summary.FileWriter('./log_dir/eval')        

        self.saver, self.save_path = self.load_cpk(self.global_step, None)

        

    def build_model(self, mode='train', testdata_dir='./data/val', traindata_dir='./data/train',batch_size=8, learning_rate1=0.001, learning_rate2=0.001):
        if mode=='train':
            tf.reset_default_graph()
            dataset, self.num_classes = creat_dataset_fromdir(traindata_dir, batch_size)
            testdataset,_ = creat_dataset_fromdir(testdata_dir, batch_size, isTrain = False)

            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

            images, labels = iterator.get_next()
            self.train_init_op = iterator.make_initializer(dataset)
            self.test_init_op = iterator.make_initializer(testdataset)

            self.build_model_train(images, labels, learning_rate1, learning_rate2, is_training=True)
            self.global_init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()

        elif mode=='test':
            tf.reset_default_graph()
            testdataset, self.num_classes = creat_dataset_fromdir(testdata_dir, batch_size, is_training=False)     
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            self.images, labels = iterator.get_next()
            self.test_init_op = iterator.make_initializer(testdataset)
            self.logits, self.end_points, self.global_step = self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)
            self.bulid_acc_base(labels)
            tf.get_default_graph.finalize()

        elif mode=='eval':
            tf.reset_default_graph()
            testdataset, self.num_classes = creat_dataset_fromdir(testdata_dir, batch_size, is_training=False)
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            self.images, labels = iterator.get_next()
            self.logits, self.end_points, self.global_step = self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)        
            tf.get_default_graph.finalize()
















        











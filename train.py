import tensorflow as tf
import os
import resource
model = __import__("model")
MyNASNetModel = model.MyNASNetModel
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 8
train_dir = 'data/train'
val_dir = 'data/val'

learning_rate1 = 1e-1
learning_rate2 = 1e-3

mymode = MyNASNetModel('./ckpt/model.ckpt')

mymode.build_model('train',val_dir,train_dir,batch_size,learning_rate1,learning_rate2)

def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 / 2, hard))


memory_limit()
num_epochs1 = 20
num_epochs2 = 200
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(mymode.global_init)

    step=0
    step = mymode.load_cpk(mymode.global_step,sess,1,mymode.saver,mymode.save_path)

    print(step)

    if step==0:
        mymode.init_fn(sess)
        for epoch in range(num_epochs1):
            print('start epoch {}/{}'.format(epoch+1,num_epochs1))
            sess.run(mymode.train_init_op)
            while True:
                try:    
                    step += 1
                    acc, accuracy_top_5,summary, _ = sess.run([mymode.accuracy, mymode.accuracy_top_5, mymode.merged, mymode.last_train_op])        
                    mymode.train_writer.add_summary(summary, step)
                    if step %100 ==0:
                        print('step:{} train1 acc:{},{}'.format(step,acc,accuracy_top_5))
                except tf.errors.OutOfRangeError:
                    print('train1:',epoch,'ok')
                    mymode.saver.save(sess, mymode.save_path+'/mynasnet.ckpt',global_step=mymode.global_step.eval())
                    break


            sess.run(mymode.step_init)

        for epoch in range(num_epochs2):
            print('start epoch {}/{}'.format(epoch+1,num_epochs2)) 
            sess.run(mymode.train_init_op)
            while True:
                try:    
                    step += 1
                    acc,summary, _ = sess.run([mymode.accuracy, mymode.merged, mymode.full_train_op])        
                    mymode.train_writer.add_summary(summary, step)
                    if step %100 ==0:
                        print('step:{} train2 acc:{},{}'.format(step,acc))
                except tf.errors.OutOfRangeError:
                    print('train2:',epoch,'ok')
                    mymode.saver.save(sess, mymode.save_path+'/mynasnet.ckpt',global_step=mymode.global_step.eval())
                    break





















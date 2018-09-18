'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
import tensorflow as tf



class NetworkConfig():

    def __init__(self):

        self.lrate =0.1
        self.num_batch=None
        self.training_epochs=10
        self.display_epoch = 1


        # specifically for DNN
        self.n_ins = 10
        self.hidden_layers_sizes = [10,10]
        self.n_outs = 10

        self.n_ins_aux = 10
        self.hidden_layers_sizes_aux = [10,10]
        self.n_outs_aux = 10

        self.sub_layer_num=3

        self.data=[]

        train_X=np.random.random((10, 10))
        train_Y=np.random.random((10, 10))
        self.inputs=train_X
        self.target=train_Y

        #files
        self.tr_input_file='path_train'
        self.dt_input_file='path_val'

    def load_file(self,fname):
        #load data
        with gzip.open(fname, 'rb') as f:
           data = cPickle.load(f)
        self.data=data
        #num of batch
        self.num_batch = len(self.data)


class Model(object):
    def __init__(self, cfg=None):
        '''n_hidden: number of hidden states
           p_keep_ff: forward keep probability
           p_keep_rc: recurrent keep probability'''
        self.cfg = cfg
        self.n_ins = cfg.n_ins; self.n_outs = cfg.n_outs
        self.hidden_layers_sizes = cfg.hidden_layers_sizes
        self.hidden_layers_number = len(self.hidden_layers_sizes)
    
        self.n_ins_aux = cfg.n_ins_aux; self.n_outs_aux = cfg.n_outs_aux
        self.hidden_layers_sizes_aux = cfg.hidden_layers_sizes_aux
        self.hidden_layers_number_aux = len(self.hidden_layers_sizes_aux)

        self.sub_layer_num=cfg.sub_layer_num

    def aux_net(self,x):

        for i in xrange(self.hidden_layers_number_aux):
            if i == 0:
                input_size =self.n_ins_aux
                layer_input = x
            else:
                input_size = self.hidden_layers_sizes_aux[i - 1]
                layer_input = output
            layname="auxlay{0}".format(i)
            with tf.variable_scope(layname) as scope:
                output = fc_layer(layer_input,input_size, self.hidden_layers_sizes_aux[i],layer_name=str(i))
        layname="auxlay{0}".format(i+1)
        with tf.variable_scope(layname) as scope:
             output = fc_layer_linear(output,self.hidden_layers_sizes_aux[-1], self.n_outs_aux,layer_name=str(i + 1))
        outputs=tf.reduce_mean(output, 0)
        return outputs
    def wei_layer(self, x,aux_out,n_in,n_out,lay_num):
        '''The structure of the network'''

        for i in xrange(self.sub_layer_num):
            layname="lay{0}_{1}".format(lay_num,i)
            with tf.variable_scope(layname) as scope:
                out= fc_layer(x,n_in, n_out,layer_name=str(i))
            if i==0:
                output=out*aux_out[i]
            else:
                output=out*aux_out[i]+output
        return output
    def main_net(self,x,aux_out):

        for i in xrange(self.hidden_layers_number):
            if i == 0:
                input_size =self.n_ins
                layer_input = x
            else:
                input_size = self.hidden_layers_sizes[i - 1]
                layer_input = output
            if i==1: 
                output=self.wei_layer(layer_input,aux_out,input_size, self.hidden_layers_sizes[i],i)
            else:
                layname="lay{0}".format(i)
                with tf.variable_scope(layname) as scope:
                     output = fc_layer(layer_input,input_size, self.hidden_layers_sizes[i],layer_name=str(i))
        layname="lay{0}".format(i+1)
        with tf.variable_scope(layname) as scope:
             output = fc_layer_linear(output,self.hidden_layers_sizes[-1], self.n_outs,layer_name=str(i + 1))
        return output
    
    
    def loss(self, output, Y):
        '''Defining the loss function'''
        cost = tf.reduce_sum(tf.pow(output-Y, 2))
        return cost 


def fc_layer(input, in_size,size, layer_name,activation=tf.nn.relu):

    w = tf.Variable(tf.truncated_normal([in_size, size]),
                    name="W" + layer_name)
    b = tf.Variable(tf.constant(0.1, shape=[size]),
                    name="b" + layer_name)
    act = activation(tf.matmul(input, w) + b, name="relu")
    return act
def fc_layer_linear(input, in_size,size, layer_name):

    w = tf.Variable(tf.truncated_normal([in_size, size]),
                    name="W" + layer_name)
    b = tf.Variable(tf.constant(0.1, shape=[size]),
                    name="b" + layer_name)
    act = tf.matmul(input, w) + b
    return act

def train_sgd_fine(train_op,cfg,cost,sess):
    fname=cfg.tr_input_file
    num_batch=10
    avg_cost = 0.
    tot_batch=0
    for i in range(1):
        for batch_index in range(num_batch):
            batch_x, batch_y=cfg.inputs,cfg.target
            _, loss = sess.run([train_op, cost],feed_dict={X: batch_x, Y: batch_y}) 
            avg_cost += loss / num_batch
    return avg_cost

def val_sgd_fine(cfg,cost,sess):
    fname=cfg.dt_input_file
    num_batch=10
    avg_cost = 0.
    tot_batch=0
    for i in range(1):
        for batch_index in range(num_batch):
            batch_x, batch_y=cfg.inputs,cfg.target
            loss= sess.run(cost,feed_dict={X: batch_x, Y: batch_y}) 
            avg_cost += loss / num_batch
    return avg_cost

if __name__ == '__main__':
# Construct model
    cfg=NetworkConfig()
    X = tf.placeholder("float",name='InputData')
    Y = tf.placeholder("float",name='TargetData')
    #cfg=NetworkConfig()
    dnn=Model(cfg=cfg)
    spk_wei=dnn.aux_net(X)
    output=dnn.main_net(X,spk_wei)
    cost = dnn.loss(output,Y)

    #this is finetune function
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(cost)

    #train_X=np.random.random((10, 10))
    #train_Y=np.random.random((10, 10))
    tf.summary.scalar("loss", cost)
    init = tf.global_variables_initializer()
    merged_summary_op = tf.summary.merge_all()

    # Start training
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('sum', graph=tf.get_default_graph())

        # Training cycle
        for epoch in range(cfg.training_epochs):
            avg_cost= train_sgd_fine(train_op,cfg,cost,sess)
            avg_cost_val = val_sgd_fine(cfg,cost,sess)

            with open("./fine_val_mean_log.txt", "a") as f:
                 f.write(str(epoch)+','+str(avg_cost_val)+'\n')

            if epoch==0:
                min_loss=avg_cost_val
            if avg_cost_val < min_loss:
                min_loss = avg_cost_val  # store your best error so far
               # saver.save(sess, "path/model.ckpt", global_step=epoch)

                       
            if (epoch+1) % cfg.display_epoch == 0:
                print("Step " + str(epoch) + ", Train_Loss= " + "{:.4f}".format(avg_cost)+ ", Val_Loss= " + "{:.4f}".format(avg_cost_val))
        print("Optimization Finished!")

        print("Run the command line:\n" \
          "--> tensorboard --logdir=sum " "Then open http://0.0.0.0:6006/ into your web browser")
    #with tf.Session() as sess: 
    #    saver = tf.train.import_meta_graph('/path/model-5.meta')
     #   saver.restore(sess,"/path/model.ckpt-5")
    #    test_error = sess.run(cost, feed_dict={X: batch_x, Y: batch_y})
    #    print("test loss " + str(test_error))

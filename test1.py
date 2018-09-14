'''
Class Model: model for new network
'''
import numpy as np
import tensorflow as tf


# from ln_lstm import LayerNormalizedLSTMCell
# from bnlstm import BNLSTMCell

class NetworkConfig():

    def __init__(self):

        self.model_type = 'DNN'

        self.batch_size = 256
        self.momentum = 0.5
        self.lrate =0.1
        self.activation=tf.nn.relu

        # specifically for DNN
        self.n_ins = 10
        self.hidden_layers_sizes = [10,10]
        self.n_outs = 10

        self.n_ins_aux = 10
        self.hidden_layers_sizes_aux = [10,10]
        self.n_outs_aux = 10

        self.sub_layer_num=3
def fc_layer(input, in_size,size, layer_name,activation=tf.nn.relu):

    w = tf.Variable(tf.truncated_normal([in_size, size]),
                    name="W" + layer_name)
    b = tf.Variable(tf.constant(0.1, shape=[size]),
                    name="b" + layer_name)
    act = tf.nn.relu(tf.matmul(input, w) + b, name="relu")
    return act

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
             output = fc_layer(output,self.hidden_layers_sizes_aux[-1], self.n_outs_aux,layer_name=str(i + 1))
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
             output = fc_layer(output,self.hidden_layers_sizes[-1], self.n_outs,layer_name=str(i + 1))
        return output
    
    
    def loss(self, output, Y):
        '''Defining the loss function'''
        cost = tf.reduce_sum(tf.pow(output-Y, 2))
        return cost 

    def train(self, loss, lr):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        # optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(
            zip(gradients, v))
        return train_op

if __name__ == '__main__':
# Construct model
    X = tf.placeholder("float",name='InputData')
    Y = tf.placeholder("float",name='TargetData')
    cfg=NetworkConfig()
    dnn=Model(cfg=cfg)
    spk_wei=dnn.aux_net(X)
    output=dnn.main_net(X,spk_wei)
    cost = dnn.loss(output,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(cost)

    train_X=np.random.random((10, 10))
    train_Y=np.random.random((10, 10))
    tf.summary.scalar("loss", cost)

    init = tf.global_variables_initializer()

    merged_summary_op = tf.summary.merge_all()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        summary_writer = tf.summary.FileWriter('sum', graph=tf.get_default_graph())
        for step in range(1, 10):
          batch_x, batch_y = train_X,train_Y
        # Run optimization op (backprop)
          _,loss,summary_str=sess.run([train_op,cost,merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
          print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))
          summary_writer.add_summary(summary_str, step)
          print("Optimization Finished!")

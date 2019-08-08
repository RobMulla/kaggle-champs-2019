import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from sklearn.utils import shuffle

import os

# Make sure tf 2.0 alpha has been installed
print(tf.__version__)

#is it using the gpu?
is_gpu = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

print('Running GPU is {}'.format(is_gpu))

##########################
##### LOAD DATA ##########
##########################

datadir = "../data/"

nodes_train     = np.load(datadir + "champs-basic-graph/nodes_train.npz" )['arr_0']
in_edges_train  = np.load(datadir + "champs-basic-graph/in_edges_train.npz")['arr_0']
out_edges_train = np.load(datadir + "champs-basic-graph/out_edges_train.npz" )['arr_0']

nodes_test     = np.load(datadir + "champs-basic-graph/nodes_test.npz" )['arr_0']
in_edges_test  = np.load(datadir + "champs-basic-graph/in_edges_test.npz")['arr_0']

out_labels = out_edges_train.reshape(-1,out_edges_train.shape[1]*out_edges_train.shape[2],1)
in_edges_train = in_edges_train.reshape(-1,in_edges_train.shape[1]*in_edges_train.shape[2],in_edges_train.shape[3])
in_edges_test  = in_edges_test.reshape(-1,in_edges_test.shape[1]*in_edges_test.shape[2],in_edges_test.shape[3])

nodes_train, in_edges_train, out_labels = shuffle(nodes_train, in_edges_train, out_labels)

### FUNCTIONS FOR MODEL

class Message_Passer_NNM(tf.keras.layers.Layer):
    def __init__(self, node_dim):
        super(Message_Passer_NNM, self).__init__()
        self.node_dim = node_dim
        self.nn = tf.keras.layers.Dense(units=self.node_dim*self.node_dim, activation=tf.nn.relu)#None)
      
    def call(self, node_j, edge_ij):
        
        # Embed the edge as a matrix
        A = self.nn(edge_ij)
        
        # Reshape so matrix mult can be done
        A = tf.reshape(A, [-1, self.node_dim, self.node_dim])
        node_j = tf.reshape(node_j, [-1, self.node_dim, 1])
        
        # Multiply edge matrix by node and shape into message list
        messages = tf.linalg.matmul(A, node_j)
        messages = tf.reshape(messages, [-1, tf.shape(edge_ij)[1], self.node_dim])

        return messages
    
class Message_Agg(tf.keras.layers.Layer):
    def __init__(self):
        super(Message_Agg, self).__init__()
    
    def call(self, messages):
        return tf.math.reduce_sum(messages, 2)
    
class Update_Func_GRU(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.GRU = tf.keras.layers.GRU(state_dim)
        
    def call(self, old_state, agg_messages):
    
        # Remember node dim
        n_nodes  = tf.shape(old_state)[1]
        node_dim = tf.shape(old_state)[2]
        
        # Reshape so GRU can be applied, concat so old_state and messages are in sequence
        old_state = tf.reshape(old_state, [-1, 1, tf.shape(old_state)[-1]])
        agg_messages = tf.reshape(agg_messages, [-1, 1, tf.shape(agg_messages)[-1]])
        concat = self.concat_layer([old_state, agg_messages])
        
        # Apply GRU and then reshape so it can be returned
        activation = self.GRU(concat)
        activation = tf.reshape(activation, [-1, n_nodes, node_dim])
        
        return activation

# Define the final output layer 
class Edge_Regressor(tf.keras.layers.Layer):
    def __init__(self, intermediate_dim):
        super(Edge_Regressor, self).__init__()
        self.concat_layer = tf.keras.layers.Concatenate()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=1, activation=None)

        
    def call(self, nodes, edges):
            
        # Remember node dims
        n_nodes  = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]
        
        # Tile and reshape to match edges
        state_i = tf.reshape(tf.tile(nodes, [1, 1, n_nodes]),[-1,n_nodes*n_nodes, node_dim ])
        state_j = tf.tile(nodes, [1, n_nodes, 1])
        
        # concat edges and nodes and apply MLP
        concat = self.concat_layer([state_i, edges, state_j])
        activation_1 = self.hidden_layer_1(concat)  
        activation_2 = self.hidden_layer_2(activation_1)

        return self.output_layer(activation_2)

# Define a single message passing layer
class MP_Layer(tf.keras.layers.Layer):
    def __init__(self, state_dim):
        super(MP_Layer, self).__init__(self)
        self.message_passers  = Message_Passer_NNM(node_dim = state_dim) 
        self.message_aggs    = Message_Agg()
        self.update_functions = Update_Func_GRU(state_dim = state_dim)
        
        self.state_dim = state_dim         
        self.batch_norm = tf.keras.layers.BatchNormalization() 

    def call(self, nodes, edges, mask):

        nodes
        edges
        
        n_nodes  = tf.shape(nodes)[1]
        node_dim = tf.shape(nodes)[2]
        
        state_j = tf.tile(nodes, [1, n_nodes, 1])

        messages  = self.message_passers(state_j, edges)

        # Do this to ignore messages from non-existant nodes
        masked =  tf.math.multiply(messages, mask)
        
        masked = tf.reshape(masked, [tf.shape(messages)[0], n_nodes, n_nodes, node_dim])

        agg_m = self.message_aggs(masked)
        
        updated_nodes = self.update_functions(nodes, agg_m)
        
        nodes_out = updated_nodes
        # Batch norm seems not to work. 
        #nodes_out = self.batch_norm(updated_nodes)
        
        return nodes_out

adj_input = tf.keras.Input(shape=(None,), name='adj_input')
nod_input = tf.keras.Input(shape=(None,), name='nod_input')
class MPNN(tf.keras.Model):
    def __init__(self, out_int_dim, state_dim, T):
        super(MPNN, self).__init__(self)   
        self.T = T
        self.embed = tf.keras.layers.Dense(units=state_dim, activation=tf.nn.relu)
        self.MP = MP_Layer( state_dim)     
        self.edge_regressor  = Edge_Regressor(out_int_dim)

        
    def call(self, inputs =  [adj_input, nod_input]):
      
      
        nodes            = inputs['nod_input']
        edges            = inputs['adj_input']

        # Get distances, and create mask wherever 0 (i.e. non-existant nodes)
        # This also masks node self-interactions...
        # This assumes distance is last
        len_edges = tf.shape(edges)[-1]
        
        _, x = tf.split(edges, [len_edges -1, 1], 2)
        mask =  tf.where(tf.equal(x, 0), x, tf.ones_like(x))
        
        # Embed node to be of the chosen node dimension (you can also just pad)
        nodes = self.embed(nodes) 
        
        # Run the T message passing steps
        for mp in range(self.T):
            nodes =  self.MP(nodes, edges, mask)
        
        # Regress the output values
        con_edges = self.edge_regressor(nodes, edges)
        
        
        return con_edges
        
######### METRICS FUNCTIONS #############
def mse(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(nums, preds)))


    return reconstruction_error

def log_mse(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.math.log(tf.reduce_mean(tf.square(tf.subtract(nums, preds))))


    return reconstruction_error

def mae(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)


    reconstruction_error = tf.reduce_mean(tf.abs(tf.subtract(nums, preds)))


    return reconstruction_error

def log_mae(orig , preds):
 
    # Mask values for which no scalar coupling exists
    mask  = tf.where(tf.equal(orig, 0), orig, tf.ones_like(orig))

    nums  = tf.boolean_mask(orig,  mask)
    preds = tf.boolean_mask(preds,  mask)

    reconstruction_error = tf.math.log(tf.reduce_mean(tf.abs(tf.subtract(nums, preds))))

    return reconstruction_error

######### SETUP MODEL RUN

learning_rate = 0.001
def step_decay(epoch):
    initial_lrate = learning_rate
    drop = 0.1
    epochs_drop = 15.0
    lrate = initial_lrate * np.power(drop,  
           np.floor((epoch)/epochs_drop))
    tf.print("Learning rate: ", lrate)
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 15, restore_best_weights=True)

#lrate  =  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                              patience=5, min_lr=0.00001, verbose = 1)

opt = tf.optimizers.Adam(learning_rate=learning_rate)

mpnn = MPNN(out_int_dim = 512, state_dim = 128, T = 5)
mpnn.compile(opt, log_mse, metrics = [mae, log_mae])

train_size = int(len(out_labels)*0.8)
batch_size = 10
epochs = 50
KERNAL_NUMBER = 'MPNN-003'

mpnn.call({'adj_input' : in_edges_train[:10], 'nod_input': nodes_train[:10]})

######### RUN THE MODEL

filepath="weights-improvement-{epoch:02d}-{val_mae:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')

log_dir="../logs/tf/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

mpnn.fit({'adj_input' : in_edges_train[:train_size],
          'nod_input': nodes_train[:train_size]},
         y = out_labels[:train_size],
         batch_size = batch_size,
         epochs = epochs, 
         callbacks = [lrate, stop_early, checkpoint, tensorboard_callback],
         use_multiprocessing = True,
         initial_epoch = 0,
         verbose = 2, 
         validation_data = ({'adj_input' : in_edges_train[train_size:],
                             'nod_input': nodes_train[train_size:]},
                            out_labels[train_size:])
        )

###### MAKE PREDICTIONS

preds = mpnn.predict({'adj_input' : in_edges_test, 'nod_input': nodes_test})

np.save(f"preds_kernel-{KERNAL_NUMBER}.npy" , preds)

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

test_group = test.groupby('molecule_name')

scale_min  = train['scalar_coupling_constant'].min()
scale_max = train['scalar_coupling_constant'].max()
scale_mid = (scale_max + scale_min)/2
scale_norm = scale_max - scale_mid

def make_outs(test_group, preds):
    i = 0
    x = np.array([])
    for test_gp, preds in zip(test_group, preds):
        if (not i%1000):
            print(i)

        gp = test_gp[1]
        
        x = np.append(x, (preds[gp['atom_index_0'].values, gp['atom_index_1'].values] + preds[gp['atom_index_1'].values, gp['atom_index_0'].values])/2.0)
        
        i = i+1
    return x

max_size = 29
preds = preds.reshape((-1,max_size, max_size))

out_unscaled = make_outs(test_group, preds)

test['scalar_coupling_constant'] = out_unscaled
test['scalar_coupling_constant'] = test['scalar_coupling_constant']*scale_norm + scale_mid
test[['id','scalar_coupling_constant']].to_csv(f'submission_{KERNAL_NUMBER}.csv', index=False)

preds_tr = mpnn.predict({'adj_input' : in_edges_train, 'nod_input': nodes_train})

np.save(f"preds_kernel-train-{KERNAL_NUMBER}.npy" , preds_tr)

max_size = 29
preds_tr = preds_tr.reshape((-1,max_size, max_size))

train_group = train.groupby('molecule_name')

out_unscaled_tr = make_outs(train_group, preds)

print('Done')
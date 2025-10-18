from __future__ import print_function
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def basis_time_encode(inputs, num_units, time_dim, expand_dim, scope='basis_time_kernal', reuse=None, return_weight=False):
    '''Mercer's time encoding

    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      time_dim: integer, number of dimention for time embedding
      expand_dim: degree of frequency expansion
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
      return_weight: bool, if true return both embeddings and frequency
    
    Returns:
      A 3d float tensor which embeds the input or 
      A tuple with one 3d float tensor (embeddings) and 2d float tensor (frequency)
    '''
    
    # inputs: [N, max_len]
    
    with tf.variable_scope('basis_time_kernal'):
        expand_input = tf.tile(tf.expand_dims(inputs, 2), [1, 1, time_dim]) # [N, max_len, time_dim]
        
        init_period_base = np.linspace(0, 8, time_dim)
        init_period_base = init_period_base.astype(np.float32)
        period_var = tf.get_variable('time_cos_freq', 
                                   dtype=tf.float32, 
                                   initializer = tf.constant(init_period_base))
        period_var = 10.0 ** period_var
        period_var = tf.tile(tf.expand_dims(period_var, 1), [1, expand_dim]) #[time_dim] -> [time_dim, 1] -> [time_dim, expand_dim]
        expand_coef = tf.cast(tf.reshape(tf.range(expand_dim) + 1, [1, -1]), tf.float32)
        
        freq_var = 1 / period_var
        freq_var = freq_var * expand_coef
        
        basis_expan_var = tf.get_variable('basis_expan_var', shape = [time_dim, 2*expand_dim], initializer=tf.glorot_uniform_initializer())
        
        basis_expan_var_bias = tf.get_variable('basis_expan_var_bias', shape = [time_dim], initializer=tf.zeros_initializer) #initializer=tf.glorot_uniform_initializer())


        sin_enc = tf.sin(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(freq_var, 0),0)))
                    
        cos_enc = tf.cos(tf.multiply(tf.expand_dims(expand_input,-1), tf.expand_dims(tf.expand_dims(freq_var, 0),0)))

        time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1), tf.expand_dims(tf.expand_dims(basis_expan_var,0),0))
        
        time_enc = tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias,0),0))

    if return_weight:
        return time_enc, freq_var
    return time_enc
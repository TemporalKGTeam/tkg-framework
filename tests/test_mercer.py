import tensorflow as tf
import numpy as np

def basis_time_encode_flex(inputs, num_units, time_dim, expand_dim, scope='basis_time_kernal', reuse=None,
                           return_weight=False):
    '''One version of the Mercer's time encoding
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
        expand_input = tf.tile(tf.expand_dims(inputs, 2), [1, 1, time_dim])  # [N, max_len, time_dim]

        init_const = np.array(
            [1.0 / d * np.array([1e8 ** (i / (time_dim - 1)) * 2 * np.pi for i in range(time_dim)]) for d in
             range(1, expand_dim + 1)]).T.astype(np.float32)

        freq_var = tf.get_variable('time_enc_freq_var', dtype=tf.float32, initializer=tf.constant(init_const))

        basis_expan_var = tf.get_variable('basis_expan_var', shape=[time_dim, 2 * expand_dim],
                                          initializer=tf.glorot_uniform_initializer())

        basis_expan_var_bias = tf.get_variable('basis_expan_var_bias', shape=[time_dim],
                                               initializer=tf.zeros_initializer)  # initializer=tf.glorot_uniform_initializer())

        inv_freq_var = tf.divide(tf.ones_like(freq_var), freq_var)
        sin_enc = tf.sin(
            tf.multiply(tf.expand_dims(expand_input, -1), tf.expand_dims(tf.expand_dims(inv_freq_var, 0), 0)))

        cos_enc = tf.cos(
            tf.multiply(tf.expand_dims(expand_input, -1), tf.expand_dims(tf.expand_dims(inv_freq_var, 0), 0)))

        time_enc = tf.multiply(tf.concat([sin_enc, cos_enc], axis=-1),
                               tf.expand_dims(tf.expand_dims(basis_expan_var, 0), 0))

        time_enc = tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias, 0), 0))

        # time_enc = tf.nn.l2_normalize(tf.add(tf.reduce_sum(time_enc, -1), tf.expand_dims(tf.expand_dims(basis_expan_var_bias,0),0)))
    if return_weight:
        return time_enc, freq_var
    return time_enc


def rand_time_encode(inputs, num_units, scope='rand_time_kernal', reuse=None, min_w=0, max_w=8):
    '''Bochner time encoding with uniformly random sampled frequencies
    Args:
      inputs: A 2d float32 tensor with shate of [N, max_len]
      num_units: An integer for the number of dimensions
      scope: string, scope for tensorflow variables
      reuse: bool, if true the layer could be reused
      min_w: float, min(log10(period))
      max_w: float, max(log10(period))

    Returns:
      A 3d float tensor which embeds the input
    '''
    assert (num_units % 2 == 0)
    effe_numits = num_units // 2
    # with tf.variable_scope(scope, reuse=reuse):
    sampled_freq = tf.random_uniform(
        [effe_numits],
        minval=10 ** min_w,
        maxval=10 ** max_w,
        dtype=tf.float32,
        seed=None,
        name=None
    )
    sampled_freq = tf.ones_like(sampled_freq) / sampled_freq
    sampled_freq = tf.contrib.framework.sort(sampled_freq)
    expand_input = tf.tile(tf.expand_dims(inputs, 2), (1, 1, effe_numits))
    cos_feat = tf.sin(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))
    sin_feat = tf.cos(tf.multiply(expand_input, tf.reshape(sampled_freq, [1, 1, effe_numits])))

    output = tf.concat([cos_feat, sin_feat], axis=2)  #= [N, max_len, num_units]
    return output


if __name__ == '__main__':
    inputs = tf.convert_to_tensor(np.random.random((64, 8)))

    time_enc = rand_time_encode(inputs, 128)

    print(time_enc)
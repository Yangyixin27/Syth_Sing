import tensorflow as tf
import math
import numpy as np
import pdb


# epsilon = 1e-9
def norm_pi(out_pi):
  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.subtract(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.multiply(normalize_pi, out_pi)
  return out_pi

def norm_data(x):
  """
  x has shape of (n, 60)
  """
  min_val = np.amin(x, axis = 0)
  max_val = np.amax(x, axis = 0)

  return (x - min_val) / (max_val - min_val)

def get_sigma(scale, skewness, gamma_s):
  sigma1 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 0)
  sigma2 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 1)
  sigma3 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 2)
  sigma4 = scale * tf.exp((gamma_s * tf.abs(skewness) - 1) * 3)
  return sigma1, sigma2, sigma3, sigma4

def get_mu(location, sigma1, sigma2, sigma3, sigma4, skewness, gamma_u):
  mu1 = location + sigma1 * gamma_u * skewness
  mu2 = location + sigma1 * gamma_u * skewness + sigma2 * gamma_u * skewness
  mu3 = location + sigma1 * gamma_u * skewness + sigma2 * gamma_u * skewness + sigma3 * gamma_u * skewness
  mu4 = location + sigma1 * gamma_u * skewness + sigma2 * gamma_u * skewness + sigma3 * gamma_u * skewness + sigma4 * gamma_u * skewness
  return mu1, mu2, mu3, mu4

def get_w(skewness, shape, gamma_w):
  # calculate denominater
  den = tf.zeros(tf.shape(skewness))
  for i in range(4):
    den += tf.pow(skewness, tf.constant(2.0 * i)) * \
            tf.pow(shape, tf.constant(i, dtype = tf.float32)) * \
            tf.pow(gamma_w, tf.constant(i, dtype = tf.float32))   # cast power term to float

  w1 = tf.div(tf.pow(skewness, tf.constant(2.0 * 0)) * tf.pow(shape, tf.constant(0.0)) * tf.pow(gamma_w, tf.constant(0.0)), den)
  w2 = tf.div(tf.pow(skewness, tf.constant(2.0 * 1)) * tf.pow(shape, tf.constant(1.0)) * tf.pow(gamma_w, tf.constant(1.0)), den)
  w3 = tf.div(tf.pow(skewness, tf.constant(2.0 * 2)) * tf.pow(shape, tf.constant(2.0)) * tf.pow(gamma_w, tf.constant(2.0)), den)
  w4 = tf.div(tf.pow(skewness, tf.constant(2.0 * 3)) * tf.pow(shape, tf.constant(3.0)) * tf.pow(gamma_w, tf.constant(3.0)), den)
  return w1, w2, w3, w4

def four_to_twelve_mapping(out_a0, out_a1, out_a2, out_a3, 
                            gamma_u = tf.constant(1.6), 
                            gamma_s = tf.constant(1.1), 
                            gamma_w = tf.constant(1/1.75)):
  """
  conversion from a0,a1,a2,a3 to mu0-mu3, sigma0-sigma3, w0-w3. All are tensors
  """
  # location, scale, skewness, shape are all shape of (1, n, 60)
  location = 2 * tf.sigmoid(out_a0) - 1
  scale =  (2/255) * tf.exp(4 * tf.sigmoid(out_a1))
  skewness = 2 * tf.sigmoid(out_a2) - 1 
  shape = 2 * tf.sigmoid(out_a3)

  sigma1, sigma2, sigma3, sigma4 = get_sigma(scale, skewness, gamma_s = gamma_s)
  mu1, mu2, mu3, mu4 = get_mu(location, sigma1, sigma2, sigma3, sigma4, skewness, gamma_u = gamma_u)
  w1, w2, w3, w4 = get_w(skewness, shape, gamma_w = gamma_w)

  return mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3, sigma4, w1, w2, w3, w4

def get_mixture_coef(output):
  """
  output is the output of wavenet, has shape of (1, n, CMG_channels)
  return 12 matrix, each of them being a matrix of shape (1, n, CMG_channels)
  """

  # out_pi, out_sigma, out_mu = tf.split(value=output, num_or_size_splits=3, axis=-1)
  # mu1, mu2, mu3, mu4 = tf.split(value=out_mu, num_or_size_splits=4, axis=-1)
  # sigma1, sigma2, sigma3, sigma4 = tf.split(value=out_sigma, num_or_size_splits=4, axis=-1)
  # pi1, pi2, pi3, pi4 = tf.split(value=out_pi, num_or_size_splits=4, axis=-1)

  # out_a0 = output[:,:,0::4]
  # out_a1 = output[:,:,1::4]
  # out_a2 = output[:,:,2::4]
  # out_a3 = output[:,:,3::4]

  # # Normalize pi's so that each pi matrix has rows sum up to one
  # pi1 = norm_pi(pi1)
  # pi2 = norm_pi(pi2)
  # pi3 = norm_pi(pi3)
  # pi4 = norm_pi(pi4)

  # # normalize sigma's so that each sigma is positive
  # sigma1 = tf.exp(sigma1)
  # sigma2 = tf.exp(sigma2)
  # sigma3 = tf.exp(sigma3)
  # sigma4 = tf.exp(sigma4)

  out_a0, out_a1, out_a2, out_a3 = tf.split(output, num_or_size_splits=4, axis=-1)

  return four_to_twelve_mapping(out_a0, out_a1, out_a2, out_a3)

def tf_normal(y, mu, sigma):
  oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
  result = tf.subtract(y, mu)
  result = tf.multiply(result,tf.reciprocal(sigma))
  result = -tf.square(result)/2
  return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

def temp_control(mu1, mu2, mu3, mu4, 
                sigma1, sigma2, sigma3, sigma4, 
                w1, w2, w3, w4, 
                tau):
  mu_bar = mu1 * w1 + mu2 * w2 + mu3 * w3 + mu4 * w4
  mu1_hat = mu1 + (mu_bar - mu1) * (1 - tau)
  mu2_hat = mu2 + (mu_bar - mu2) * (1 - tau)
  mu3_hat = mu3 + (mu_bar - mu3) * (1 - tau)
  mu4_hat = mu4 + (mu_bar - mu4) * (1 - tau)
  sigma1_hat = sigma1 * tf.sqrt(tau)
  sigma2_hat = sigma2 * tf.sqrt(tau)
  sigma3_hat = sigma3 * tf.sqrt(tau)
  sigma4_hat = sigma4 * tf.sqrt(tau)

  return mu1_hat, mu2_hat, mu3_hat, mu4_hat, sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, w1, w2,w3, w4

def get_lossfunc(mu1_hat, mu2_hat, mu3_hat, mu4_hat, 
                sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, 
                w1, w2, w3, w4, 
                y): 
  # result1 = tf.multiply(tf_normal(y, mu1, sigma1), w1)
  # result2 = tf.multiply(tf_normal(y, mu2, sigma2), w2)
  # result3 = tf.multiply(tf_normal(y, mu3, sigma3), w3)
  # result4 = tf.multiply(tf_normal(y, mu4, sigma4), w4)
  # result = result1 + result2 + result3 + result4

  d1 = tf.distributions.Normal(loc = mu1_hat, scale = sigma1_hat)
  d2 = tf.distributions.Normal(loc = mu2_hat, scale = sigma2_hat)
  d3 = tf.distributions.Normal(loc = mu3_hat, scale = sigma3_hat)
  d4 = tf.distributions.Normal(loc = mu4_hat, scale = sigma4_hat)

  prob = w1 * d1.prob(y) + w2 * d2.prob(y) + w3 * d3.prob(y) + w4 * d4.prob(y)


  # prob = prob + epsilon

  logprob = -tf.log(prob)
  result = tf.reduce_sum(logprob, axis = -1)
  return tf.reduce_mean(result, axis = -1)

if __name__ == "__main__":
  NHIDDEN = 64
  # STDEV = 0.5
  MFSC_dim = 60
  # KMIX = 4 # number of mixtures
  # NOUT = MFSC_dim * KMIX * 3 # pi, mu, stdev
  NOUT = MFSC_dim * 4
  NEPOCH = 1000

  x = tf.placeholder(dtype=tf.float32, shape=[1, None, MFSC_dim], name="x")
  y = tf.placeholder(dtype=tf.float32, shape=[1, None, MFSC_dim], name="y")


  initializer = tf.contrib.layers.xavier_initializer_conv2d()


  # Wh = tf.Variable(tf.random_normal([MFSC_dim,NHIDDEN], stddev=STDEV, dtype=tf.float32))
  # bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

  Wh = tf.Variable(initializer(shape=[1, MFSC_dim, NHIDDEN]), dtype=tf.float32)
  bh = tf.Variable(initializer(shape=[NHIDDEN]), dtype=tf.float32)


  Wo = tf.Variable(initializer(shape=[1, NHIDDEN, NOUT]), dtype=tf.float32)
  bo = tf.Variable(initializer(shape=[NOUT]), dtype=tf.float32)

  hidden_layer = tf.nn.tanh(tf.nn.conv1d(x, Wh, stride=1, padding="SAME") + bh)
  output = tf.nn.conv1d(hidden_layer,Wo, stride=1, padding="SAME") + bo

  # mu1, mu2, mu3, mu4, sigma1, sigma2, sigma3, sigma4, pi1, pi2, pi3, pi4 = get_mixture_coef(output)
  ################### debugging ########################
  # out_a0 = output[:,:,0::4]
  # out_a1 = output[:,:,1::4]
  # out_a2 = output[:,:,2::4]
  # out_a3 = output[:,:,3::4]

  out_a0, out_a1, out_a2, out_a3 = tf.split(output, num_or_size_splits=4, axis=-1)

  location = 2 * tf.sigmoid(out_a0) - 1
  scale =  (2/255) * tf.exp(4 * tf.sigmoid(out_a1))
  skewness = 2 * tf.sigmoid(out_a2) - 1 
  shape = 2 * tf.sigmoid(out_a3)

  sigma1, sigma2, sigma3, sigma4 = get_sigma(scale, skewness, gamma_s = tf.constant(1.1))
  mu1, mu2, mu3, mu4 = get_mu(location, sigma1, sigma2, sigma3, sigma4, skewness, gamma_u = tf.constant(1.6))
  w1, w2, w3, w4 = get_w(skewness, shape, gamma_w = tf.constant(1/1.75))
  # piecewise linear tau
  tau = np.concatenate((np.array([0.05] * 3),
                        np.linspace(0.05, 0.5, 6),
                        np.array([0.5] * 51)),
                        axis = 0)
  tau = tf.constant(tau, dtype=tf.float32)

  # temperature control
  mu_bar = mu1 * w1 + mu2 * w2 + mu3 * w3 + mu4 * w4
  mu1_hat = mu1 + (mu_bar - mu1) * (1 - tau)
  mu2_hat = mu2 + (mu_bar - mu2) * (1 - tau)
  mu3_hat = mu3 + (mu_bar - mu3) * (1 - tau)
  mu4_hat = mu4 + (mu_bar - mu4) * (1 - tau)
  sigma1_hat = sigma1 * tf.sqrt(tau)
  sigma2_hat = sigma2 * tf.sqrt(tau)
  sigma3_hat = sigma3 * tf.sqrt(tau)
  sigma4_hat = sigma4 * tf.sqrt(tau)

  # result1 = tf.multiply(tf_normal(y, mu1_hat, sigma1_hat), w1)
  # result2 = tf.multiply(tf_normal(y, mu2_hat, sigma2_hat), w2)
  # result3 = tf.multiply(tf_normal(y, mu3_hat, sigma3_hat), w3)
  # result4 = tf.multiply(tf_normal(y, mu4_hat, sigma4_hat), w4)
  # result = result1 + result2 + result3 + result4
  #######################################################
  # mix_gaussian = tf_normal(y, )
  # lossfunc = get_lossfunc(mu1_hat, mu2_hat, mu3_hat, mu4_hat, sigma1_hat, sigma2_hat, sigma3_hat, sigma4_hat, w1, w2, w3, w4, y)
  d1 = tf.distributions.Normal(loc = mu1_hat, scale = sigma1_hat)
  d2 = tf.distributions.Normal(loc = mu2_hat, scale = sigma2_hat)
  d3 = tf.distributions.Normal(loc = mu3_hat, scale = sigma3_hat)
  d4 = tf.distributions.Normal(loc = mu4_hat, scale = sigma4_hat)

  prob = w1 * d1.prob(y) + w2 * d2.prob(y) + w3 * d3.prob(y) + w4 * d4.prob(y)

  epsilon = tf.constant(1e-9, dtype=tf.float32)
  prob = prob + epsilon

  logprob = -tf.log(prob)
  result = tf.reduce_sum(logprob, axis = -1)
  result = tf.reduce_mean(result, axis = -1)

  train_op = tf.train.AdamOptimizer(learning_rate=0.000005).minimize(result)


  # loss = np.zeros(NEPOCH) # store the training progress here.
  x_data = np.load("mfsc_nitech_jp_song070_f001_003.npy")[:64, :]
  x_data = norm_data(x_data)
  x_data = x_data.reshape((1, x_data.shape[0], x_data.shape[1]))
  y_data = x_data

  # pdb.set_trace()

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # print (sum(sess.run(pi1, feed_dict={x:x_data})[0]))

    # print sess.run()

    for i in range(NEPOCH):
      # pdb.set_trace()
      sess.run(train_op,feed_dict={x: x_data, y: y_data})
      # pdb.set_trace()
      print ("out_a0:", sess.run(out_a0,feed_dict={x: x_data, y: y_data}))
      print ("out_a1:", sess.run(out_a1,feed_dict={x: x_data, y: y_data}))
      print ("out_a2:", sess.run(out_a2,feed_dict={x: x_data, y: y_data}))
      print ("out_a3:", sess.run(out_a3,feed_dict={x: x_data, y: y_data}))
      print ("Wh:", sess.run(Wh,feed_dict={x: x_data, y: y_data}))
      print ("bh:", sess.run(bh,feed_dict={x: x_data, y: y_data}))
      print ("Wo:", sess.run(Wo,feed_dict={x: x_data, y: y_data}))
      print ("bo:", sess.run(bo,feed_dict={x: x_data, y: y_data}))
      if i % 20 ==0:
        # pdb.set_trace()
        loss_val = sess.run(result, feed_dict={x: x_data, y: y_data})
        print ("step = {}, loss = {}".format(i+1, loss_val))

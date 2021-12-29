import numpy as np
import tensorflow as tf
import sklearn as sk
import matplotlib.pyplot as plt

#############################################
#This bit is loading the data, which is somewhat manual, change to match your applcation
x_train = np.load('x_train_256.npy')
y_train = np.load('y_train_256.npy')
x_train, y_train = sk.utils.shuffle(x_train, y_train)

pos_ind = [num for num,i in enumerate(y_train) if  i == 1]
x_train_pos = x_train[pos_ind]

neg_ind = [num for num,i in enumerate(y_train) if  i == 0]
x_train_neg = x_train[neg_ind]
#############################################

# define the standalone discriminator model
def make_disc_model(n_inputs=256):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
  model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
 
# define the standalone generator model
def make_gen_model(latent_dim, n_outputs=256):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
  model.add(tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'))
  model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(tf.keras.layers.Dense(n_outputs, activation='linear'))
  return model


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n, latent_dim)
	return x_input

def make_gan_model(disc_model, gen_modl):
  # make weights in the discriminator not trainable
	disc_model.trainable = False
	# connect them
	model = tf.keras.Sequential()
	# add generator
	model.add(gen_modl)
	# add the discriminator
	model.add(disc_model)
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

# use the generator to generate n fake examples, with class labels
def generate_fake_sample(generator, latent_dim, n):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = np.zeros((n, 1))
	return X, y

def generate_real_sample(n):
  inds = []
  for i in range(n):
    inds.append(int(np.random.rand()*len(x_train_pos)))
  X = x_train_pos[inds]
  # generate class labels
  y = np.ones((n, 1))
  return X, y

def eval_perf(epoch, gen_model, disc_model, latent_dim = 5):
    n = 100
    x_real, y_real = generate_real_sample(n)
    loss_real, acc_real = disc_model.evaluate(x_real, y_real, verbose=0)
    latent_vec = np.random.randn(n, latent_dim)
    x_fake, y_fake = gen_model.predict(latent_vec), np.zeros(n)
    loss_fake, acc_fake = disc_model.evaluate(x_fake, y_fake, verbose=0)
    print(n)
    print(x_fake.shape)
    print("{}, {}, {}".format(epoch, acc_real, acc_fake))
    #if epoch%500 == 0:
    plt.subplot(211)
    plt.plot(x_fake[10], label = 'real')
    plt.subplot(212)
    plt.psd(x_fake[10], Fs = 256, label = 'fake')
    plt.legend()
    plt.show()

def train(disc_model, gen_model, gan_model, epochs, latent_dim = 5, n_eval=100):
    batch_size = 128
    half_batch = int(batch_size / 2)
	# manually enumerate epochs
    for i in range(epochs):
      # prepare real samples
      x_real, y_real = generate_real_sample(half_batch)
      # prepare fake examples
      x_fake, y_fake = generate_fake_sample(gen_model, latent_dim, half_batch)
      # update discriminator
      disc_model.train_on_batch(x_real, y_real)
      disc_model.train_on_batch(x_fake, y_fake)
      # prepare points in latent space as input for the generator
      x_gan = generate_latent_points(latent_dim, batch_size)
      # create inverted labels for the fake samples
      y_gan = np.ones((batch_size, 1))
      # update the generator via the discriminator's error
      gan_model.train_on_batch(x_gan, y_gan)
      # evaluate the model every n_eval epochs
      if (i+1) % n_eval == 0:
        eval_perf(i, gen_model, disc_model, latent_dim)
            
latent_dim = 5
# create the discriminator
discriminator = make_disc_model(256)
# create the generator
generator = make_gen_model(latent_dim, 256)
# create the gan
gan_model = make_gan_model(discriminator, generator)
# train model
train(discriminator, generator, gan_model, latent_dim = 5, epochs = 50000)
        

from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv1D
import tensorflow as tf
import zinc_grammar as G

# helper variables in Keras format for parsing the grammar
masks_K = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 277
DIM = G.D


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    z_mean, z_log_var = args           # mean and log of variance of Q(z|X)
    batch_size = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, dim))
    # return: sampled latent vector
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class MoleculeVAE:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.AE = None
        self.encoderMV = None

    def create(self, charset, max_length=MAX_LEN, latent_rep_size=2, weights_file=None):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        _, z = self.encode(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(encoded_input,
                             self.decode(encoded_input, latent_rep_size, max_length, charset_length))

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self.decode(x1, latent_rep_size, max_length)
        self.AE = Model(x1,
                        self.decode(z1, latent_rep_size, max_length, charset_length))

        # for obtaining mean and log variance of encoding distribution
        x2 = Input(shape=(max_length, charset_length))
        (z_m, z_l_v) = self.encode_mean_var(x2, latent_rep_size)
        self.encoderMV = Model(input=x2, output=[z_m, z_l_v])

        if weights_file:
            self.AE.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)
            self.encoderMV.load_weights(weights_file, by_name=True)

        self.AE.compile(optimizer='Adam',
                        loss=vae_loss,
                        metrics=['accuracy'])

    @staticmethod
    def encode_mean_var(self, x, latent_rep_size):
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='conv_1')(x)
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='conv_2')(h)
        h = Conv1D(filters=10, kernel_size=11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return z_mean, z_log_var

    @staticmethod
    def encode(x, latent_rep_size, max_length):
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='conv_1')(x)
        h = Conv1D(filters=9, kernel_size=9, activation='relu', name='conv_2')(h)
        h = Conv1D(filters=10, kernel_size=11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        # this function is the main change.
        # essentially we mask the training data so that we are only allowed to apply
        #  future rules based on the current non-terminal
        def conditional(x_true, x_pred):
            most_likely = K.argmax(x_true)
            most_likely = tf.reshape(most_likely, [-1]) # flatten most_likely
            ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely), 1) # index ind_of_ind with res
            ix2 = tf.cast(ix2, tf.int32) # cast indices as ints
            M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
            M3 = tf.reshape(M2, [-1, MAX_LEN, DIM]) # reshape them
            P2 = tf.matmul(K.exp(x_pred), M3) # apply them to the exp-predictions
            P2 = tf.div(P2, K.sum(P2, axis=-1, keepdims=True)) # normalize predictions
            return P2

        def vae_loss(x, x_decoded_mean):
            x_decoded_mean = conditional(x, x_decoded_mean) # we add this new function to the loss
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        return vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

    @staticmethod
    def decode(z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences=True, name='gru_1')(h)
        h = GRU(501, return_sequences=True, name='gru_2')(h)
        h = GRU(501, return_sequences=True, name='gru_3')(h)
        # don't do SoftMax, we do this in the loss now
        return TimeDistributed(Dense(charset_length), name='decoded_mean')(h)

    def save(self, filename):
        self.AE.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=2, max_length=MAX_LEN):
        self.create(charset, max_length=max_length, weights_file=weights_file, latent_rep_size=latent_rep_size)

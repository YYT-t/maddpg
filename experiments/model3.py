import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.disable_eager_execution()
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.optimizers import Adam
from tensorflow.compat.v1.keras.layers import Lambda, Input, Dense, Concatenate, Reshape, Add, Multiply
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.losses import categorical_crossentropy
from tensorflow.compat.v1.keras.layers import Activation

beta_2 = 0.05


def my_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + beta_2 * categorical_crossentropy(y_pred, y_pred)


def intrisic_eoi(dim, num_classes):
    In = Input(shape=(dim,))
    X = Dense(64, activation='relu')(In)
    X = Dense(64, activation='relu')(X)
    X = Dense(num_classes, activation='softmax')(X)
    model = Model(In, X)
    # model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001))
    model.compile(loss=my_loss, optimizer=Adam(0.0001))
    return model


def build_batch_eoi(dim, eoi_net, n_ant):
    In = []
    R = []
    for i in range(n_ant):
        In.append(Input(shape=(dim,)))
        R.append(Lambda(lambda x: K.reshape(x[:, i], (-1, 1)))(eoi_net(In[i])))

    model = Model(In, R)

    return model


def build_q_net(num_features, n_actions):
    O = Input(shape=(num_features,))
    h = Dense(64, activation='relu')(O)
    h = Dense(64, activation='relu')(h)
    V = Dense(n_actions)(h)
    model = Model(O, V)
    return model


def build_critic(num_features, n_actions):
    O = Input(shape=(num_features,))
    h = Dense(64, activation='relu')(O)
    h = Dense(64, activation='relu')(h)
    V = Dense(n_actions)(h)

    model = Model(O, V)
    return model


def build_mixer(n_ant, state_space):
    I1 = Input(shape=(n_ant,))
    I2 = Input(shape=(state_space,))

    W1 = Dense(n_ant * 64)(I2)
    W1 = Lambda(lambda x: K.abs(x))(W1)
    W1 = Reshape((n_ant, 64))(W1)
    b1 = Dense(64)(I2)

    W2 = Dense(64)(I2)
    W2 = Lambda(lambda x: K.abs(x))(W2)
    W2 = Reshape((64, 1))(W2)
    b2 = Dense(1)(I2)

    h = Lambda(lambda x: K.batch_dot(x[0], x[1]))([I1, W1])
    h = Add()([h, b1])
    h = Activation('relu')(h)
    q_total = Lambda(lambda x: K.batch_dot(x[0], x[1]))([h, W2])
    q_total = Add()([q_total, b2])

    model = Model([I1, I2], q_total)
    return model


def build_Q_tot(observation_space, n_actions, state_space, n_ant, q_nets, mixer):
    O = []
    for i in range(n_ant):
        O.append(Input(shape=(observation_space,)))
    A = []
    for i in range(n_ant):
        A.append(Input(shape=(n_actions,)))
    S = Input(shape=(state_space,))

    q_values = []
    for i in range(n_ant):
        q_value = q_nets(O[i])
        q_values.append(Lambda(lambda x: K.reshape(K.sum(x, axis=1), (-1, 1)))(Multiply()([A[i], q_value])))
    q_values = Concatenate(axis=1)(q_values)
    q_total = mixer([q_values, S])

    model = Model(O + A + [S], q_total)
    return model


def build_acting(num_features, actors, n_ant):
    Inputs = []
    for i in range(n_ant):
        Inputs.append(Input(shape=(num_features,)))

    actions = []
    for i in range(n_ant):
        actions.append(actors(Inputs[i]))

    return K.function(Inputs, actions)


def build_batch_q(dim, n_ant, actors, critics):
    O = []
    for i in range(n_ant):
        O.append(Input(shape=(dim,)))

    Q_small = []
    Q_in = []
    for i in range(n_ant):
        Q_small.append(actors(O[i]))
        Q_in.append(critics[i]([O[i]]))

    return K.function(O, Q_small + Q_in)


class Agent(object):
    def __init__(self, sess, observation_space, n_actions, state_space, n_ant, alpha):
        super(Agent, self).__init__()
        self.sess = sess
        self.observation_space = observation_space
        self.n_actions = n_actions
        self.n_ant = n_ant
        self.state_space = state_space
        self.alpha = alpha
        self.critics = []
        self.critics_tar = []
        K.set_session(sess)

        self.q_nets = build_q_net(self.observation_space, self.n_actions)#输入obs输出所有actions的估值
        for i in range(self.n_ant):
            self.critics.append(build_critic(self.observation_space, self.n_actions))#结构一样但是损失函数不一样所以不一样？
        self.mixer = build_mixer(self.n_ant, self.state_space)
        self.Q_tot = build_Q_tot(self.observation_space, self.n_actions, self.state_space, self.n_ant, self.q_nets,
                                 self.mixer)
        self.acting = build_acting(self.observation_space, self.q_nets, self.n_ant)
        self.batch_q = build_batch_q(self.observation_space, self.n_ant, self.q_nets, self.critics)

        self.q_nets_tar = build_q_net(self.observation_space, self.n_actions)
        for i in range(self.n_ant):
            self.critics_tar.append(build_critic(self.observation_space, self.n_actions))
        self.mixer_tar = build_mixer(self.n_ant, self.state_space)
        self.Q_tot_tar = build_Q_tot(self.observation_space, self.n_actions, self.state_space, self.n_ant,
                                     self.q_nets_tar, self.mixer_tar)
        self.batch_q_tar = build_batch_q(self.observation_space, self.n_ant, self.q_nets_tar, self.critics_tar)

        self.label = tf.placeholder(tf.float32, [None, 1])
        self.action_masks = []
        for i in range(self.n_ant):
            self.action_masks.append(tf.placeholder(tf.float32, [None, n_actions]))
        self.optimize = []

        loss_q = tf.reduce_mean((self.label - self.Q_tot.output) ** 2)
        loss_i = tf.reduce_mean([tf.nn.softmax_cross_entropy_with_logits(
            logits=self.q_nets(self.Q_tot.inputs[i]) - 9e15 * (1 - self.action_masks[i]),
            labels=tf.nn.softmax(self.critics[i](self.Q_tot.inputs[i]) - 9e15 * (1 - self.action_masks[i]))) for i in
                                 range(self.n_ant)])

        self.optimize.append(tf.train.AdamOptimizer(0.0005).minimize(loss_q + self.alpha * loss_i,
                                                                     var_list=self.q_nets.trainable_weights))
        self.optimize.append(
            tf.train.AdamOptimizer(0.0005).minimize(tf.reduce_mean((self.label - self.Q_tot.output) ** 2),
                                                    var_list=self.mixer.trainable_weights))
        self.opt_qmix = tf.group(self.optimize)

        self.opt_critic = []
        self.label_critic = []
        for i in range(self.n_ant):
            self.label_critic.append(tf.placeholder(tf.float32, [None, n_actions]))
            self.opt_critic.append(tf.train.AdamOptimizer(0.0005).minimize(
                tf.reduce_mean((self.label_critic[i] - self.critics[i].outputs[0]) ** 2),
                var_list=self.critics[i].trainable_weights))
        self.opt_critic = tf.group(self.opt_critic)

        self.sess.run(tf.global_variables_initializer())

    def train_qmix(self, O, A, S, mask, label):

        dict1 = {}
        for i in range(self.n_ant):
            dict1[self.Q_tot.inputs[i]] = O[i]
            dict1[self.Q_tot.inputs[i + self.n_ant]] = A[i]
            dict1[self.action_masks[i]] = mask[i]
        dict1[self.Q_tot.inputs[2 * self.n_ant]] = S
        dict1[self.label] = label
        self.sess.run(self.opt_qmix, feed_dict=dict1)

    def train_critics(self, X, label):

        dict1 = {}
        for i in range(self.n_ant):
            dict1[self.critics[i].inputs[0]] = X[i]
            dict1[self.label_critic[i]] = label[i]
        self.sess.run(self.opt_critic, feed_dict=dict1)

    def update(self):
        weights = self.Q_tot.get_weights()
        self.Q_tot_tar.set_weights(weights)
        for i in range(self.n_ant):
            weights = self.critics[i].get_weights()
            self.critics_tar[i].set_weights(weights)

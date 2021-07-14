import tensorflow as tf

class CTCClassifier(tf.keras.Model):
    def __init__(self, cfg, name="CTCClassifier", **kwargs):
        super(CTCClassifier, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(cfg.filter_size, cfg.kernel_size, [2, 2], padding='valid', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(cfg.filter_size, cfg.kernel_size, [2, 2], padding='valid', activation=tf.nn.relu)
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.lstm1 = tf.keras.layers.LSTM(cfg.dense_size, return_sequences=True)
        self.fc1 = tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(cfg.class_size)
    
    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.lstm1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class TaskSpecificNetwork(tf.keras.Model):
    def __init__(self, cfg, name="TaskSpecificNetwork", **kwargs):
        super(TaskSpecificNetwork, self).__init__(name=name, **kwargs)
        self.fc1 = tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(cfg.class_size)
    
    def call(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class ExtractorNetwork(tf.keras.Model):
    def __init__(self, cfg, name="Extractor", **kwargs):
        super(ExtractorNetwork, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(cfg.filter_size, cfg.kernel_size, [2, 2], padding='valid', activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(cfg.filter_size, cfg.kernel_size, [2, 2], padding='valid', activation=tf.nn.relu)
        self.flatten = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.fc1 = tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu)
    
    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out

class ContextNetwork(tf.keras.Model):
    def __init__(self, cfg, pred_size, name="ContextNetwork", **kwargs):
        super(ContextNetwork, self).__init__(name=name, **kwargs)
        self.fc1 = tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu)
        self.fc3 = [tf.keras.layers.Dense(cfg.dense_size, activation=tf.nn.relu) for _ in range(pred_size)]
        self._pred_size = pred_size
    
    def call(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        outputs = []
        for i in range(len(self.fc3)):
            outputs.append(self.fc3[i](out[:, :- self._pred_size, :]))
        return outputs
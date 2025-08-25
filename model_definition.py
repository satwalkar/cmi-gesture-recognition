# model_definition.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback #, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, BatchNormalization, Conv1D, Multiply,
    Concatenate, Bidirectional, Activation, LayerNormalization,
    MultiHeadAttention, Embedding, Reshape
)
from tensorflow.keras.regularizers import l2

import config

# ==========================================================
# Keras Tuner Availability Check & Fallback
# ==========================================================
try:
    import keras_tuner as kt
    _KERAS_TUNER_AVAILABLE = True
    print("✅ Keras Tuner is available.")
except ImportError:
    print("⚠️ Warning: 'keras_tuner' library not found. Hyperparameter Optimization will be skipped.")
    _KERAS_TUNER_AVAILABLE = False

    # Create a dummy kt object and HyperParameters class to prevent NameErrors
    # This allows the model_builder function signature to remain consistent.
    class DummyHP:
        def __init__(self):
            pass
        def Choice(self, name, values, **kwargs):
            return values[0]
        def Int(self, name, min_value, **kwargs):
            return min_value
        def Float(self, name, min_value, **kwargs):
            return min_value
        def Boolean(self, name, **kwargs):
            return False
        def Fixed(self, name, value, **kwargs):
            return value

    class KTDummy:
        def __init__(self):
            self.HyperParameters = DummyHP

    kt = KTDummy()

# ==========================================================

# --- ALL CUSTOM LAYERS AND BLOCKS ---
# --- Helper Layer for Instance Normalization (from a user request) ---
# These components are self-contained and can be moved here directly.
class InstanceNormalization(tf.keras.layers.Layer):
    """
    A custom Instance Normalization Layer for Keras.
    This layer normalizes the input along the features axis for each sample in the batch.
    It is useful when the style or content of an individual instance is important, as it
    is independent of other samples in the batch.
    """
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )
        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.gamma * normalized + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config

class LRLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Check if learning_rate is a callable schedule or a direct tensor/value
        if callable(self.model.optimizer.learning_rate):
            current_lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            current_lr = self.model.optimizer.learning_rate # It's already the scalar value

        print(f"Epoch {epoch + 1}: Current LR = {current_lr.numpy()}")

# --- NEW: Custom Layer for Sum Aggregation ---
# This replaces the final problematic Lambda layer in the attention block.
# A custom layer is fully serializable and robust for saving/loading models.
class SumOverTimeDimension(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Sums the input tensor along the second axis (the time dimension).
        # Input shape: (batch, time, features) -> Output shape: (batch, features)
        return tf.keras.backend.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        # The output shape will be the input shape with the time dimension removed.
        return (input_shape[0], input_shape[2])

    def get_config(self):
        # No configuration needed for this simple layer.
        base_config = super().get_config()
        return base_config

# --- 6. Model Architecture (Multi-Input with Attention) ---
# Positional Embedding for Transformer (B.3)
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        # Using a fixed seed for reproducibility for embedding layers
        self.position_embedding = Embedding(input_dim=sequence_length, output_dim=output_dim, embeddings_initializer=tf.keras.initializers.RandomNormal(seed=config.RANDOM_STATE))

    def call(self, inputs):
        length = tf.shape(inputs)[-2]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
        })
        return config

# Transformer Encoder Block (B.3 & B.4)
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, kernel_initializer=None, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.kernel_initializer = kernel_initializer # Store it

        # Using a fixed seed for reproducibility for MultiHeadAttention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, kernel_initializer=self.kernel_initializer)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=l2(rate), kernel_initializer=self.kernel_initializer),
            Dense(embed_dim, kernel_regularizer=l2(rate), kernel_initializer=self.kernel_initializer),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
            # Add initializer to config for model saving
            "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
        })
        return config

# --- MODIFICATION: Replaced fragile Lambda layers with robust Reshape layers ---
def attention_block(inputs):
    """
    Custom attention block. Uses standard Keras layers for robust saving and loading.
    """
    # This block generates weights based on the input sequence
    x = Dense(inputs.shape[-1], activation='tanh', use_bias=False, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config.RANDOM_STATE))(inputs)
    x = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=config.RANDOM_STATE))(x)

    # Squeeze the last dimension (from shape (None, 128, 1) to (None, 128))
    # using a Reshape layer, which is fully serializable.
    x = Reshape((-1,))(x)

    weights = Activation('softmax')(x)

    # Expand the last dimension back (from (None, 128) to (None, 128, 1))
    # for element-wise multiplication.
    weights = Reshape((-1, 1))(weights)

    # This multiplies the weights with the original inputs
    context = Multiply()([inputs, weights])

	# This Lambda layer is fine as it was already defined with an output_shape.
    # This aggregates the weighted inputs across the time dimension (axis=1).
    # We explicitly provide the output_shape to prevent a NotImplementedError when loading the saved model.
    output = SumOverTimeDimension()(context)

    return output

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class OneCycleLR(tf.keras.callbacks.Callback):
    """
    A custom Keras callback to implement the 1cycle learning rate policy.
    """
    def __init__(self, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super(OneCycleLR, self).__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_up = int(self.total_steps * self.pct_start)
        self.step_down = self.total_steps - self.step_up

    def on_train_begin(self, logs=None):
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        self.lrs = []
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_batch_end(self, batch, logs=None):
        step = tf.keras.backend.get_value(self.model.optimizer.iterations)
        if step <= self.step_up:
            # Ramp up phase
            new_lr = self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.step_up)
        else:
            # Ramp down phase
            progress = (step - self.step_up) / self.step_down
            new_lr = self.max_lr - (self.max_lr - self.min_lr) * progress
        
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.lrs.append(new_lr)

def model_builder(hp: kt.HyperParameters, ts_shape: tuple, demo_shape: tuple, num_classes: int):
    """
    Builds the Keras model with a flexible, config-driven architecture.
    This version has a clean, unified structure.
    """
    # --- 1. Define Full Hyperparameter Search Space (ONCE) ---
    l2_reg_strength = hp.Choice('l2_reg', values=[1e-4, 1e-5, 1e-3], default=1e-4)
    conv1d_filters_1 = hp.Choice('conv1d_filters_1', values=[32, 48, 64], default=32)
    conv1d_kernel_1 = hp.Choice('conv1d_kernel_1', values=[5, 3], default=5)
    conv1d_filters_2 = hp.Choice('conv1d_filters_2', values=[64, 96, 128], default=64)
    conv1d_kernel_2 = hp.Choice('conv1d_kernel_2', values=[3, 5], default=3)
    
    # GRU-specific HPs
    gru_units_1 = hp.Int('gru_units_1', min_value=32, max_value=96, step=32, default=64)
    gru_units_2 = hp.Int('gru_units_2', min_value=16, max_value=64, step=16, default=32)
    ts_dropout_rate = hp.Float('ts_dropout_rate', min_value=0.3, max_value=0.5, step=0.1, default=0.4)
    
    # Transformer-specific HPs
    transformer_num_heads = hp.Choice('transformer_num_heads', values=[4, 8, 2], default=4)
    transformer_ff_dim = hp.Int('transformer_ff_dim', min_value=128, max_value=384, step=128, default=256)
    transformer_dropout_rate = hp.Float('transformer_dropout_rate', min_value=0.2, max_value=0.4, step=0.1, default=0.3)
    
    # Final classifier HPs
    dense_units_demo = hp.Choice('dense_units_demo', values=[16, 32], default=16)
    combined_dense_units = hp.Choice('combined_dense_units', values=[64, 96, 128], default=64)
    combined_dropout_rate = hp.Float('combined_dropout_rate', min_value=0.3, max_value=0.5, step=0.1, default=0.4)
    
    # --- 2. Define Common Arguments ---
    l2_reg = l2(l2_reg_strength)
    initializer = tf.keras.initializers.GlorotUniform(seed=config.RANDOM_STATE)

    # --- 3. Build Model Architecture ---
    ts_input = Input(shape=ts_shape, name='time_series_input')
    x = Conv1D(conv1d_filters_1, conv1d_kernel_1, activation='relu', padding='same', kernel_regularizer=l2_reg, kernel_initializer=initializer)(ts_input)
    x = InstanceNormalization()(x)
    x = Conv1D(conv1d_filters_2, conv1d_kernel_2, activation='relu', padding='same', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)

    # --- Core Sequential Branch ---
    if config.MODEL_TYPE == 'gru':
        x = Bidirectional(GRU(gru_units_1, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        x = Bidirectional(GRU(gru_units_2, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        
        if config.ATTENTION_TYPE == 'simple':
            x = attention_block(x)
        else: # bahdanau
            query_gru = GRU(gru_units_1, return_sequences=False)(x)
            x, _ = BahdanauAttention(128)(x, query_gru)
        ts_features = Dropout(ts_dropout_rate)(x)

    elif config.MODEL_TYPE == 'transformer':
        x = PositionalEmbedding(sequence_length=config.MAX_SEQUENCE_LENGTH, output_dim=conv1d_filters_2)(x)
        x = TransformerEncoder(embed_dim=conv1d_filters_2, num_heads=transformer_num_heads,
                               ff_dim=transformer_ff_dim, rate=transformer_dropout_rate,
                               kernel_initializer=initializer)(x)
        x = attention_block(x)
        ts_features = Dropout(combined_dropout_rate)(x)

    elif config.MODEL_TYPE == 'hybrid':
        x = Bidirectional(GRU(gru_units_1, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        x = TransformerEncoder(embed_dim=gru_units_1*2, num_heads=transformer_num_heads,
                               ff_dim=transformer_ff_dim, rate=transformer_dropout_rate,
                               kernel_initializer=initializer)(x)
        x = attention_block(x)
        ts_features = Dropout(combined_dropout_rate)(x)

    else:
        raise ValueError(f"Invalid MODEL_TYPE in config: '{config.MODEL_TYPE}'. Choose 'gru', 'transformer', or 'hybrid'.")

    # --- Final Classifier Branch ---
    if config.ENABLE_DEMOGRAPHICS:
        demo_input = Input(shape=demo_shape, name='demographics_input')
        y = Dense(dense_units_demo, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(demo_input)
        y = BatchNormalization()(y)
        combined = Concatenate()([ts_features, y])
        final_classifier_input = combined
        model_inputs = [ts_input, demo_input]
    else:
        final_classifier_input = ts_features
        model_inputs = [ts_input]

    z = Dropout(combined_dropout_rate)(final_classifier_input)
    z = Dense(combined_dense_units, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(z)
    output_layer = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=model_inputs, outputs=output_layer)
    
    # Return the UNCOMPILED model. Compilation is handled in the training stage.
    return model

"""    
def model_builder(hp: kt.HyperParameters, ts_shape: tuple, demo_shape: tuple, num_classes: int):
    
    # Builds the Keras model for Hyperparameter Optimization (B.1).
    # Incorporates Transformer block and Multi-Head Attention (B.3, B.4).
    # This function now dynamically chooses the normalization layer based on `NORMALIZATION_LAYER_TYPE`.

    # Hyperparameters for the model architecture

    # L2 Regularization: Centered around 1e-4
    l2_reg_strength = hp.Choice('l2_reg', values=[1e-4, 1e-5, 1e-3], default=1e-4)

    # Conv1D Layers: Searching in a tight range around 32 and 64 filters

    #conv1d_filters_1 = hp.Int('conv1d_filters_1', min_value=32, max_value=128, step=32)
    conv1d_filters_1 = hp.Choice('conv1d_filters_1', values=[32, 48, 64], default=32)

    #conv1d_kernel_1 = hp.Choice('conv1d_kernel_1', values=[3, 5])
    conv1d_kernel_1 = hp.Choice('conv1d_kernel_1', values=[5, 3], default=5)

    #conv1d_filters_2 = hp.Int('conv1d_filters_2', min_value=64, max_value=256, step=64)
    conv1d_filters_2 = hp.Choice('conv1d_filters_2', values=[64, 96, 128], default=64)

    #conv1d_kernel_2 = hp.Choice('conv1d_kernel_2', values=[3, 5])
    conv1d_kernel_2 = hp.Choice('conv1d_kernel_2', values=[3, 5], default=3)

    if not config.ENABLE_TRANSFORMER_BLOCK:
        # GRU Layers: Searching around 64 and 32 units

        #gru_units_1 = hp.Int('gru_units_1', min_value=64, max_value=256, step=64)
        gru_units_1 = hp.Int('gru_units_1', min_value=32, max_value=96, step=32, default=64)

        #gru_units_2 = hp.Int('gru_units_2', min_value=32, max_value=128, step=32)
        gru_units_2 = hp.Int('gru_units_2', min_value=16, max_value=64, step=16, default=32)

        # Dropout Rates: Searching around 0.4
        #ts_dropout_rate = hp.Float('ts_dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        ts_dropout_rate = hp.Float('ts_dropout_rate', min_value=0.3, max_value=0.5, step=0.1, default=0.4)

        #consider next
        #ts_dropout_rate = hp.Float('ts_dropout_rate', min_value=0.4, max_value=0.6, step=0.1)

    else: # Transformer related HPs
        # The embedding dimension for the transformer must match the last dimension of its input.
        # This is the number of features per time step after the CNN layers.
        # We'll set the transformer_embed_dim to be equal to conv1d_filters_2
        transformer_embed_dim = conv1d_filters_2 # Fixed to match previous layer output

        # We can still tune num_heads and ff_dim
        transformer_num_heads = hp.Choice('transformer_num_heads', values=[4, 8, 2], default=4)
        transformer_ff_dim = hp.Int('transformer_ff_dim', min_value=128, max_value=384, step=128, default=256)
        transformer_dropout_rate = hp.Float('transformer_dropout_rate', min_value=0.2, max_value=0.4, step=0.1, default=0.3)

    # Dense Layer Units: Searching around 16 and a reasonable default of 64
    #dense_units_demo = hp.Int('dense_units_demo', min_value=16, max_value=64, step=16)
    dense_units_demo = hp.Choice('dense_units_demo', values=[16, 32], default=16)

    #combined_dense_units = hp.Int('combined_dense_units', min_value=64, max_value=256, step=64)
    combined_dense_units = hp.Choice('combined_dense_units', values=[64, 96, 128], default=64)

    # Dropout Rates: Searching around 0.4
    #combined_dropout_rate = hp.Float('combined_dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    combined_dropout_rate = hp.Float('combined_dropout_rate', min_value=0.3, max_value=0.5, step=0.1, default=0.4)

    #consider next
    #combined_dropout_rate = hp.Float('combined_dropout_rate', min_value=0.4, max_value=0.6, step=0.1)

    learning_rate_choice = hp.Choice('learning_rate_schedule', values=['cosine_decay', 'constant'], default='cosine_decay')

    #initial_learning_rate = hp.Float('initial_learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')
    initial_learning_rate = hp.Float('initial_learning_rate', min_value=3e-4, max_value=2e-3, sampling='log', default=5e-4)

    # Define common arguments once
    l2_reg = l2(l2_reg_strength)
    initializer = tf.keras.initializers.GlorotUniform(seed=config.RANDOM_STATE)

    # Time Series Input Branch
    ts_input = Input(shape=ts_shape, name='time_series_input')

    # --- 3. Build Initial Feature Extractor (common to all models) ---
    x = Conv1D(conv1d_filters_1, conv1d_kernel_1, activation='relu', padding='same', kernel_regularizer=l2_reg, kernel_initializer=initializer)(ts_input)

    # Use the selected normalization layer here
    x = InstanceNormalization()(x)
    x = Conv1D(conv1d_filters_2, conv1d_kernel_2, activation='relu', padding='same', kernel_regularizer=l2_reg, kernel_initializer=initializer)(x)

    # Hardcode InstanceNormalization for the time-series branch
    x = InstanceNormalization()(x)

    # --- 4. Build the Core Sequential Branch (based on MODEL_TYPE) ---
    # --- Time Series Branch: GRU, Transformer, or Hybrid  ---    
    if config.MODEL_TYPE == 'gru':
        print("Building GRU model...")
        
        # --- DEFINE GRU-specific hyperparameters HERE ---
        gru_units_1 = hp.Int('gru_units_1', min_value=32, max_value=96, step=32, default=64)
        gru_units_2 = hp.Int('gru_units_2', min_value=16, max_value=64, step=16, default=32)
        ts_dropout_rate = hp.Float('ts_dropout_rate', min_value=0.3, max_value=0.5, step=0.1, default=0.4)
        
        # Now use them to build the layers
        x = Bidirectional(GRU(gru_units_1, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        x = Bidirectional(GRU(gru_units_2, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        
        # Apply the chosen attention mechanism
        if config.ATTENTION_TYPE == 'simple':
            x = attention_block(x)
        else: # bahdanau
            # Note: Bahdanau attention needs a query vector, often the last state of another RNN
            query_gru = GRU(gru_units_1, return_sequences=False)(x)
            x, _ = BahdanauAttention(128)(x, query_gru)
        
        ts_features = Dropout(ts_dropout_rate)(x)

    elif config.MODEL_TYPE == 'transformer':
        print("Building Transformer model...")
        
        # --- DEFINE Transformer-specific hyperparameters HERE ---
        transformer_num_heads = hp.Choice('transformer_num_heads', values=[4, 8, 2], default=4)
        transformer_ff_dim = hp.Int('transformer_ff_dim', min_value=128, max_value=384, step=128, default=256)
        transformer_dropout_rate = hp.Float('transformer_dropout_rate', min_value=0.2, max_value=0.4, step=0.1, default=0.3)
        
        # Now use them to build the layers
        x = PositionalEmbedding(sequence_length=config.MAX_SEQUENCE_LENGTH, output_dim=conv1d_filters_2)(x)
        x = TransformerEncoder(embed_dim=conv1d_filters_2, num_heads=transformer_num_heads,
                               ff_dim=transformer_ff_dim, rate=transformer_dropout_rate,
                               kernel_initializer=initializer)(x)
        x = attention_block(x) # Use simple attention to get a final context vector
        ts_features = Dropout(combined_dropout_rate)(x)

    elif config.MODEL_TYPE == 'hybrid':
        print("Building Hybrid GRU-Transformer model...")
        # First, process with GRU
        x = Bidirectional(GRU(gru_units_1, return_sequences=True, kernel_regularizer=l2_reg, kernel_initializer=initializer))(x)
        x = InstanceNormalization()(x)
        # Then, feed the result to the Transformer
        x = TransformerEncoder(embed_dim=gru_units_1*2, num_heads=transformer_num_heads,
                               ff_dim=transformer_ff_dim, rate=transformer_dropout_rate,
                               kernel_initializer=initializer)(x)
        x = attention_block(x)
        ts_features = Dropout(combined_dropout_rate)(x)

    else:
        raise ValueError(f"Invalid MODEL_TYPE in config: '{config.MODEL_TYPE}'. Choose 'gru', 'transformer', or 'hybrid'.")

    # --- 5. Build the Demographics Branch and Final Classifier ---
    if config.ENABLE_DEMOGRAPHICS:
        demo_input = Input(shape=demo_shape, name='demographics_input')
        y = Dense(dense_units_demo, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(demo_input)
        y = BatchNormalization()(y)
        
        combined = Concatenate()([ts_features, y])
        final_classifier_input = combined
        model_inputs = [ts_input, demo_input]
    else:
        final_classifier_input = ts_features
        model_inputs = [ts_input]

    z = Dropout(combined_dropout_rate)(final_classifier_input)
    z = Dense(combined_dense_units, activation='relu', kernel_regularizer=l2_reg, kernel_initializer=initializer)(z)
    output_layer = Dense(num_classes, activation='softmax', name='output')(z)

    model = Model(inputs=model_inputs, outputs=output_layer)

    # Learning Rate Schedule (B.1)
    # The actual decay_steps value will be determined in run_training_pipeline
    # This just defines the *type* of schedule and initial LR for the optimizer.
    if learning_rate_choice == 'constant':
        lr_schedule_obj = initial_learning_rate
    elif learning_rate_choice == 'cosine_decay':
        # Placeholder for decay_steps in model_builder for HPO.
        # The actual steps_per_epoch will be calculated in the tuner.search or model.fit.
        # Keras Tuner will handle passing this correctly or setting up the LR schedule.
        # For non-HPO, it will be handled by run_training_pipeline.
        lr_schedule_obj = CosineDecay(initial_learning_rate, decay_steps=1000) # Dummy decay_steps for model definition # This will be overridden later.

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_obj, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Model built successfully with multiple inputs.")
    # model.summary() # Commented out to avoid verbose output during HPO
    return model
"""
    
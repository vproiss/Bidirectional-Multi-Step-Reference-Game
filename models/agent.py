import tensorflow as tf
from configs.settings import (
    SUBSET_SIZE, 
    MESSAGE_LENGTH, 
    NUM_MESSAGES, 
    EMBEDDING_SIZE, 
    COM_FEATURE_SIZE, 
    FEATURE_SIZE
)


class MessageFeatureExtractor(tf.keras.Model):
    """
    A feature extractor for messages that handles both integer-encoded and pre-embedded messages.
    This class performs an embedding lookup for integer-encoded messages, followed by a dense transformation
    and LSTM processing to capture sequential dependencies.
    """
    def __init__(self):
        super().__init__()
        # Embedding layer to convert integer-encoded messages into dense vectors.
        self.embed = tf.keras.layers.Embedding(NUM_MESSAGES, EMBEDDING_SIZE)
         # Time-distributed dense layer to transform pre-embedded message sequences.
        self.dense_transform = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(EMBEDDING_SIZE))
        # LSTM layer to capture sequential dependencies in the message sequence.
        self.lstm = tf.keras.layers.LSTM(COM_FEATURE_SIZE, return_state=False)

    @tf.function()
    def call(self, inputs):
        """
        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, message_length) for integer-encoded messages
                                or (batch_size, message_length, embedding_size) for pre-embedded messages.

        Returns:
            output (tf.Tensor): Output tensor of shape (batch_size, com_feature_size).
        """
        # Check if the input is a sequence of integer-encoded messages.
        if len(inputs.shape) == 2: 
            # If true, convert the integer-encoded messages into dense vectors using the embedding layer.
            x = self.embed(inputs)
            # Assert that the shape of the embedding output is as expected.
            tf.debugging.assert_shapes([(x, (None, MESSAGE_LENGTH, EMBEDDING_SIZE))], 
                                       message="Unexpected shape for embedding output.")
        else:
            # If the input is already a sequence of embedded messages, pass it through the dense transformation layer. 
            x = self.dense_transform(inputs)
        
        # Pass the transformed input through the LSTM layer to capture sequential dependencies.
        output = self.lstm(x)
        
        # Assert that the shape of the LSTM output is as expected.
        tf.debugging.assert_shapes([(output, (None, COM_FEATURE_SIZE))], 
                                   message="Unexpected shape for LSTM output.")
        return output
   

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Multi-head self-attention layer.
    
    This layer performs self-attention on an input sequence, dividing the input into 
    multiple heads to allow the model to jointly attend to information from different 
    representation subspaces.
    
    Attributes:
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.
        wq (Dense): The dense layer for query transformation.
        wk (Dense): The dense layer for key transformation.
        wv (Dense): The dense layer for value transformation.
        dense (Dense): The final dense layer to transform the concatenated attention outputs.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Ensure the dimensionality of the model is divisible by the number of heads.
        tf.debugging.assert_equal(
            d_model % self.num_heads, 0,
            message="The d_model must be divisible by num_heads."
        )
        
        # Initialize dense layers for query, key, value, and output transformation.
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, subset_size, d_model).
            
        Returns:
            tf.Tensor: Reshaped tensor of shape (batch_size, num_heads, subset_size, depth).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    @tf.function
    def call(self, v, k, q):
        """
        Perform multi-head self-attention.
        
        Args:
            v (tf.Tensor): Value tensor of shape (batch_size, subset_size, d_model).
            k (tf.Tensor): Key tensor of shape (batch_size, subset_size, d_model).
            q (tf.Tensor): Query tensor of shape (batch_size, subset_size, d_model).
            
        Returns:
            tf.Tensor: Output tensor of the multi-head self-attention layer (batch_size, subset_size, feature_size).
        """
        batch_size = tf.shape(q)[0]
        
        # Apply linear transformations.
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split the dimensions into multiple heads.
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Calculate the scaled dot-product attention.
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        
        # Ensure the shape of the output tensor.
        tf.debugging.assert_equal(tf.shape(scaled_attention)[1], self.num_heads,
                                  message="The number of heads in scaled attention is not equal to num_heads."
        )
        tf.debugging.assert_equal(tf.shape(attention_weights)[1], self.num_heads,
                                  message="The number of heads in attention weights is not equal to num_heads."
        )
        
        # Concatenate the heads and apply the final linear layer.
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output

    def scaled_dot_product_attention(self, q, k, v):
        """
        Calculate the scaled dot-product attention.
        
        Args:
            q (tf.Tensor): Query tensor of shape (batch_size, num_heads, subset_size, depth).
            k (tf.Tensor): Key tensor of shape (batch_size, num_heads, subset_size, depth).
            v (tf.Tensor): Value tensor of shape (batch_size, num_heads, subset_size, depth).
            
        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The output tensor and the attention weights.
        """
        # Calculate the dot product of query and key tensors.
        # The result is a tensor of shape (batch_size, num_heads, subset_size, subset_size).
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Scale the dot-product by the square root of the dimension of the keys.
        # This is done to prevent the dot-product from growing too large as the dimensionality increases.
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the softmax activation to get the attention weights.
        # The softmax is applied on the last axis (axis=-1) so that the weights sum to 1 for each sequence.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Multiply the attention weights with the value tensor to get the output.
        # The result is a tensor of shape (batch_size, num_heads, subset_size, depth).
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights


class ReferenceObjectFeatureExtractor(tf.keras.Model):
    """
    Feature extractor for reference objects.

    This class implements a feature extraction pipeline that includes a dense transformation,
    multi-head self-attention, and layer normalization.
    """
    def __init__(self, num_heads=8):
        super(ReferenceObjectFeatureExtractor, self).__init__()
        self.num_heads = num_heads
        self.dense_transform = tf.keras.layers.Dense(2048)  
        self.attention = MultiHeadSelfAttention(d_model=2048, num_heads=self.num_heads) 
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense = None # Defined in build method.
        self.output_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        """
        The build method is used to define layers that depend on the shape of the input.
        """
        feature_size = input_shape[-1]
        self.dense = tf.keras.layers.Dense(feature_size, activation='relu')
    
    @tf.function
    def call(self, inputs):
        """
        Forward pass through the feature extraction pipeline.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch_size, subset_size, feature_size).

        Returns:
            tf.Tensor: Output tensor of the feature extraction pipeline of shape (batch_size, subset_size, feature_size).
        """
        # Apply a dense transformation to the input tensor.
        transformed_inputs = self.dense_transform(inputs)

        # Pass the transformed input through the multi-head self-attention layer.
        attention_output = self.attention(transformed_inputs, transformed_inputs, transformed_inputs)

        # Normalize the sum of the attention output and the transformed input.
        x = self.attention_norm(attention_output + transformed_inputs)

        # Apply another dense transformation to the normalized tensor.
        x = self.dense(x)

        #Apply layer normalization to the output tensor.
        x = self.output_norm(x)

        return x
    

class AgentCore(tf.keras.Model):
    """
    The AgentCore class processes incoming message features and reference object features to generate a meaningful representation for the agent.
    This representation captures the temporal dependencies between the communication and the reference object features,
    using LSTM layers for sequence processing.
    """
    def __init__(self, num_lstm_layers=3):
        super(AgentCore, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.comm_transform = tf.keras.layers.Dense(FEATURE_SIZE)  
        
    def build(self, input_shapes):
        """
        The build method is used to define layers that depend on the shape of the input.
        """
        comm_shape, ref_shape = input_shapes
        lstm_units = COM_FEATURE_SIZE  
        self.lstms = [tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
                      for _ in range(self.num_lstm_layers)]
        self.dense = tf.keras.layers.Dense(COM_FEATURE_SIZE)  

    @tf.function
    def call(self, inputs):
        """
        Args:
            inputs: Tuple containing message features (shape: (batch_size, com_feature_size))
                    and reference features (shape: (batch_size, subset_size, feature_size)).

        Returns:
            output: Core output (shape: (batch_size, com_feature_size)),
                    list of hidden states, and list of cell states.
        """
        comm_features, ref_features = inputs
        
        # Validate the shape of the input tensors.
        tf.debugging.assert_shapes([
            (comm_features, (None, COM_FEATURE_SIZE)),
            (ref_features, (None, None, FEATURE_SIZE))
        ], message="Invalid shapes for inputts.")
        
        # Transform the message features.
        comm_features = tf.expand_dims(comm_features, 1)
        comm_features = self.comm_transform(comm_features) 

        # Average the reference features.
        averaged_features = tf.reduce_mean(ref_features, axis=1, keepdims=True)

        # Concatenate the transformed message features and averaged reference features.
        x = tf.concat([comm_features, averaged_features], axis=-1)
        
        # Process the concatenated features through LSTM layers.
        h_states = []
        c_states = []
        for lstm in self.lstms:
            x, h, c = lstm(x)
            h_states.append(h)
            c_states.append(c)

        # Remove unnecessary dimensions from the output tensor.
        x = tf.squeeze(x, axis=1)
        
        # Return the output tensor, list of hidden states, and list of cell states.
        return x, h_states, c_states
    

class MessageGenerator(tf.keras.Model):
    """
    The MessageGenerator class takes in the agent's core representation and generates a message to be sent to other agents.
    This is achieved using a dense layer to transform the agent's core representation into a message.
    """
    def __init__(self, lstm_units, message_length, num_messages):
        super(MessageGenerator, self).__init__()
        # An LSTM layer to capture the sequential dependencies in the input.
        # This helps in understanding the sequence of the agent's core representation.
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
        # A dense layer to produce a probability distribution over possible messages.
        # This is the final output of the MessageGenerator.
        self.dense = tf.keras.layers.Dense(num_messages, activation='softmax')
        # The length of the message to be generated.
        self.message_length = message_length
    
    @tf.function
    def call(self, inputs):
        """
        Generate a probability distribution over possible messages.

        Args:
            inputs: A tensor of shape (batch_size, feature_size) representing the input features.

        Returns:
            message_probabilities: A tensor of shape (batch_size, message_length, num_messages) representing 
            the probability distribution over possible messages.
        """
        # Expand the dimension of the input tensor to match the expected shape for LSTM.
        inputs = tf.expand_dims(inputs, 1)

        # Repeat the input tensor along the sequence axis to match the message length.
        inputs = tf.repeat(inputs, self.message_length, axis=1)

        # Pass the inputs through the LSTM layer to capture the sequential dependencies.
        lstm_output = self.lstm(inputs)

        # Pass the LSTM output through the dense layer to generate the probability distribution over possible messages.
        message_probabilities = self.dense(lstm_output)

        #Return the probability distribution over possible messages.
        return message_probabilities
    

class PredictionGenerator(tf.keras.Model):
    """
    A prediction generator that produces predictions for each object in the reference subset.
    The predictions represent the likelihood of each reference object being the target of the agent's attention.
    """
    def __init__(self, output_units=SUBSET_SIZE):
        """
        Initialize the PredictionGenerator.

        Args:
            output_units (int): The number of output units, equal to the size of the reference subset.
        """
        super(PredictionGenerator, self).__init__()
        # A dense layer to generate predictions for each object in the reference subset.
        # The sigmoid activation function maps the output to a range between 0 and 1,
        # representing probabilities.
        self.dense = tf.keras.layers.Dense(output_units, activation='sigmoid')
    
    @tf.function
    def call(self, inputs):
        """
        Generate predictions for each object in the reference subset.

        Args:
            inputs: A tensor of shape (batch_size, feature_size) representing the input features.

        Returns:
            predictions: A tensor of shape (batch_size, subset_size) representing 
            the probability distribution over reference object space.
        """
        # Get the batch size from the shape of the input tensor.
        batch_size = tf.shape(inputs)[0]
        
        # Assert that the input tensor has the correct shape.
        tf.debugging.assert_equal(tf.shape(inputs)[0], batch_size,
                                  message="Input tensor must have a batch size dimension."
        )
        tf.debugging.assert_equal(tf.shape(inputs)[1], COM_FEATURE_SIZE,
                                  message="Input tensor must have a second dimension of size COM_FEATURE_SIZE."
        )
        
        # Compute predictions using the dense layer.
        predictions = self.dense(inputs)
        
        # Assert that the output tensor has the correct shape.
        tf.debugging.assert_equal(tf.shape(predictions)[0], batch_size,
                                  message="Output tensor must have a batch size dimension."
        )
        tf.debugging.assert_equal(tf.shape(predictions)[1], self.dense.units,
                                  message="Output tensor must have a second dimension of size equal to the number of output units."
        )
        
        return predictions
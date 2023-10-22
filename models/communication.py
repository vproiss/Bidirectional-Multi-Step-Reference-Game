import tensorflow as tf
from configs.settings import (
    MESSAGE_LENGTH, 
    NUM_MESSAGES, 
    COM_FEATURE_SIZE, 
)

from models.agent import (
    MessageFeatureExtractor, 
    ReferenceObjectFeatureExtractor, 
    AgentCore, MessageGenerator, 
    PredictionGenerator
)

class CommunicationAgent(tf.keras.Model):
    """
    A CommunicationAgent model that encapsulates the process of generating 
    messages and predictions based on input features.
    """
    def __init__(self):
        super(CommunicationAgent, self).__init__()
        # Feature extractor for messages.
        self.message_feature_extractor = MessageFeatureExtractor()
        # Feature extractor for reference objects.
        self.reference_object_feature_extractor = ReferenceObjectFeatureExtractor()
        # The core processing unit of the agent.
        self.agent_core = AgentCore()
        # Generator for new messages.
        self.message_generator = MessageGenerator(COM_FEATURE_SIZE, MESSAGE_LENGTH, NUM_MESSAGES)
        # Generator for predictions.
        self.prediction_generator = PredictionGenerator()
    
    #@tf.function
    def call(self, ref_objects, messages, states=None):
        """
        The main processing pipeline of the CommunicationAgent.

        Args:
            ref_objects: A tensor representing the reference objects.
            messages: A tensor representing the messages from other agents.
            states: The initial states of the agent core (optional).
        
        Returns:
            new_messages_probabilities: A tensor representing the probabilities of the new messages.
            predictions: A tensor representing the predictions for each reference object.
            new_states: The updated states of the agent core.
        """
        # Extract features from reference objects.
        ref_obj_features = self.reference_object_feature_extractor(ref_objects)
        # Extract message features.
        message_output = self.message_feature_extractor(messages)
 
        # Assert that the shape of message_output is as expected.
        tf.debugging.assert_shapes([(message_output, ('batch_size', 'COM_FEATURE_SIZE'))], 
                                   message="Unexpected shape for message_output!"
        )
        
        # Process features through the agent core.
        core_output, h_states, c_states = self.agent_core([message_output, ref_obj_features])
        # Get the updated states of the agent core.
        new_states = [h_states, c_states]
        # Generate new messages based on the core output.
        new_messages_probabilities = self.message_generator(core_output)
         # Generate predictions based on the core output.
        predictions = self.prediction_generator(core_output)

        return new_messages_probabilities, predictions, new_states
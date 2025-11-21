import numpy as np
import tensorflow as tf


class ExpressionClassifier:
    def __init__(self, model_path, categories_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open(categories_path, "r") as file:
            content = file.readline().strip()
            self.categories = [name.capitalize() for name in content.split(",")]
            

    def predict(self, input_data: list | np.ndarray):
        # # Ensure input_data is a numpy array with the correct dtype
        input_data = np.array(input_data, dtype=np.float32)
        # Assume input_data is a single sample (1D array), expand to 2 dimensions for the model
        input_data = np.expand_dims(input_data, axis=0)

        # Check if input shape matches model's expected input shape
        expected_input_shape = self.input_details[0]["shape"]
        if input_data.shape[1] != expected_input_shape[1]:
            raise ValueError(
                f"Input data has {input_data.shape[1]} features, but model expects {expected_input_shape[1]} features."
            )

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Assuming output_data contains logits, apply softmax to get probabilities
        probabilities = tf.nn.softmax(output_data).numpy()

        # Get the predicted class index and confidence
        predicted_class_index = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)

        # Map index to category name
        predicted_category = [self.categories[i] for i in predicted_class_index]

        # return predicted_category[0], confidence[0]
        return predicted_category[0]
    


if __name__ == "__main__":
    # For test purpose
    model_path = "models/expression_classifier.tflite"
    categories_path="blendshapes/categories.csv"
    sample=[2.51023675e-06, 2.87316016e-05, 1.01197475e-05, 9.41790164e-01,
                           9.04946148e-01, 9.67411399e-01, 1.09630826e-04, 6.34157459e-06,
                           1.35863229e-05, 5.61126955e-02, 1.65914036e-02, 1.53885812e-01,
                           1.16858244e-01, 4.29587245e-01, 8.91092047e-03, 7.25111179e-03,
                           4.34758484e-01, 7.88603649e-02, 1.15144953e-01, 2.88044155e-01,
                           1.86897516e-01, 1.30558452e-02, 6.65456653e-02, 8.33365601e-04,
                           1.64610834e-03, 6.41273800e-03, 1.19670221e-04, 8.14346015e-04,
                           5.09596430e-02, 1.75022997e-03, 6.31518709e-03, 8.98747891e-03,
                           8.79504834e-04, 2.10408792e-01, 3.41309002e-03, 6.28193701e-03,
                           6.11386411e-02, 2.07623504e-02, 7.55252969e-03, 1.56928604e-06,
                           1.93851553e-02, 2.81072524e-03, 3.54258576e-03, 7.26593472e-03,
                           3.31033438e-01, 1.42615944e-01, 1.80225477e-01, 2.51664311e-01,
                           1.83176118e-04, 1.04853760e-04, 4.54242127e-06, 2.61033006e-06]
    
    classifier = ExpressionClassifier(model_path, categories_path)
    result = classifier.predict(sample)
    print(result)

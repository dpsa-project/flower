
import flwr as fl

import tensorflow as tf

if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client, dpsa
    class CifarClient(fl.client.DpsaNumPyClient):
        def get_parameters(self):  # type: ignore
            return model.get_weights()

        def dpsa_fit(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
            weights = model.get_weights()
            print("Hey, my weights have length:")
            print(len(weights))
            return true, weights, len(x_train)

        def fit(self, parameters, config):
            return None

        def evaluate(self, parameters, config):  # type: ignore
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, len(x_test), accuracy

    # Start Flower client
    fl.client.start_dpsa_numpy_client("[::]:8080", client=CifarClient())



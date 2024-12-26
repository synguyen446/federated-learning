import flwr as fl
from utils import *
import logging

X_train, X_test, y_train, y_test = load_data()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model = load_model()
        return model.get_weights()

    def fit(self, parameters, config):
        model = load_model()
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=3, batch_size=128)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model = load_model()
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
        logging.log(logging.INFO, f"ACCURACY: {accuracy} - LOSS: {loss} ")
        return loss, len(X_test), {"accuracy": accuracy}


fl.client.start_numpy_client(
    server_address="192.168.86.21:8080", client=FlowerClient())

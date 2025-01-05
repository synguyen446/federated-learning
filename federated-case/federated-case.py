import utils
from tensorflow.keras.models import load_model

_,X_test,_,y_test = utils.load_data()
model = load_model("models/model-round10.keras")

loss, accuracy = model.evaluate(X_test, y_test,batch_size=32)
print(f"Evaluation Accuracy: {accuracy:.5f}")
print(f"Evaluation Loss: {loss:.5f}")

utils.visualize(model,X_test)






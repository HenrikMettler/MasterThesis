# Source: https://rubikscode.net/2018/11/26/3-ways-to-implement-autoencoders-with-tensorflow-and-python/


from keras.layers import Dense, Input
from keras.models import Model
from keras import regularizers

class Autoencoder(object):

    def __init__(self, inputLayer_size, hiddenLayer_size, hiddenLayer_activationFunction, outputLayer_activationFunction, outputLayer_size, optimizer, loss):
        input_layer = Input(shape=(inputLayer_size,))
        hidden_input = Input(shape=(hiddenLayer_size,))
        hidden_layer = Dense(hiddenLayer_size, activation=hiddenLayer_activationFunction)(input_layer)
        output_layer = Dense(outputLayer_size, activation=outputLayer_activationFunction)(hidden_layer)

        self._autoencoder_model = Model(input_layer, output_layer)
        self._encoder_model = Model(input_layer, hidden_layer)
        tmp_decoder_layer = self._autoencoder_model.layers[-1]
        self._decoder_model = Model(hidden_input, tmp_decoder_layer(hidden_input))

        self._autoencoder_model.compile(optimizer=optimizer, loss=loss)

    def train(self, input_train, input_test, batch_size, epochs):
        self._autoencoder_model.fit(input_train,
                                    input_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_data=(
                                        input_test,
                                        input_test))

    def getEncodedImage(self, image):
        encoded_image = self._encoder_model.predict(image)
        return encoded_image

    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._decoder_model.predict(encoded_imgs)
        return decoded_image
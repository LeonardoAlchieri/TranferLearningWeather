# general purpose
from tensorflow.python.compiler.mlcompute import mlcompute
import numpy as np
from yaml import safe_load
from sys import getsizeof
from warnings import warn
# plots
import matplotlib.pyplot as plt
import seaborn as sns
# load data
import h5py
# ML packages
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (Conv2D, Dense, MaxPool2D, GlobalMaxPool2D,
                                     AveragePooling2D, Input, Flatten, Dropout,
                                     Concatenate)
import tensorflow.keras as keras
from tensorflow import executing_eagerly
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input

# this way I do not have to reshape the data
keras.backend.set_image_data_format('channels_first')
assert keras.backend.image_data_format() == 'channels_first', "Plase set keras backend as channels first."

# APPLE SPECIFIC IMPLEMENTATION
disable_eager_execution()
assert executing_eagerly() is False, "Please do not set to eager execution for this program."


# Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
mlcompute.set_mlc_device(device_name='gpu')  # Available options are 'cpu', 'gpu', and 'any'.


def transfer_learning_VGG(x_train = None, y_train = None, x_test = None, y_test = None, slicing = 15, memory_efficient = False, optimizer = 'adam', batch_size = 128, epochs = 10):
    assert (x_train is not None) and (y_train is not None) and (x_test is not None) and (y_test is not None), "Forgot to give some variables."

    if (x_train.shape[1] == 3) and (x_test.shape[1] == 3):
        print("[INFO] Loading base VGG model.")
        INPUT = (x_train[0].shape)
        new_input = Input(shape=INPUT)
        VGG_model = VGG16(include_top=False, input_tensor=new_input)
        base_model = Model(inputs = VGG_model.input, outputs = VGG_model.layers[slicing].output)
        if memory_efficient is False:
            print("[INFO] Using VGG as feature extractor and then training new model.")
            warn("This method uses more memory. If you want to save memory, set memory_efficient to True. It will instruct the program to run the feature extraction phase at training time (MUCH SLOWER.)")
            # just as a test, I shall use only the first 3 images
            x_train = base_model.predict(preprocess_input(x = x_train))
            x_test = base_model.predict(preprocess_input(x = x_test))
            # This architecture is very similar to the
            # one proposed by the researchers
            print("[INFO] Creating new model")
            model = Sequential([
                Input(shape=x_train[0].shape),
                Conv2D(16, kernel_size=(2, 2), activation='relu'),
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            print("[INFO] Compiling model.")
            model.compile(loss = 'binary_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])
            print("[INFO] Model traning.")
            history = model.fit(x_train,
                y_train,
                verbose = 1,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks = [EarlyStopping(monitor = 'val_loss',
                                            patience = 5,
                                            verbose = 1)])
            print("[INFO] Final test accuracy: %.2f" %(model.evaluate(x_test, y_test)[1]))
            print("[INFO] Training completed.")
            # joint_model = Model(inputs = base_model.input, outputs = model.output)
            # plot_model(joint_model)
            return history, model
        else:
            print("[INFO] Creating new model with sliced VGG as non-trainable layer: saves memory but much slower.")
            model = Sequential([
                Input(shape=base_model.layers[-1].output_shape[1:]),
                Conv2D(16, kernel_size=(2, 2), activation='relu'),
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            print("[INFO] Joining models")
            for layer in base_model.layers:
                layer.trainable = False
            output = model(base_model.outputs)
            joined_models = Model(base_model.inputs, output)
            print("[INFO] Compiling model.")
            joined_models.compile(loss = 'binary_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])
            print("[INFO] Model traning.")
            history = joined_models.fit(preprocess_input(x_train),
                y_train,
                verbose = 1,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(preprocess_input(x_test), y_test))
            print("[INFO] Final test accuracy: %.2f" %(joined_models.evaluate(x_test, y_test)[1]))
            print("[INFO] Training completed.")
            # joint_model = Model(inputs = base_model.input, outputs = model.output)
            # plot_model(joint_model)
            return history, joined_models
    elif (x_train.shape[1] == 1) and (x_test.shape[1] == 1):
        print("[INFO] Loading base VGG model.")
        print("[INFO] One-dimensional channel mode activated")
        INPUT = (x_train[0].shape)
        new_input = Input(shape=INPUT)
        img_conc = Concatenate(axis = 1)([new_input, new_input, new_input])
        VGG_model = VGG16(include_top=False, input_tensor=img_conc)
        base_model = Model(inputs = VGG_model.input, outputs = VGG_model.layers[15].output)
        if memory_efficient is False:
            print("[INFO] Using VGG as feature extractor and then training new model.")
            warn("This method uses more memory. If you want to save memory, set memory_efficient to True. It will instruct the program to run the feature extraction phase at training time (MUCH SLOWER.)")
            # just as a test, I shall use only the first 3 images
            x_train = base_model.predict(x = x_train)
            x_test = base_model.predict(x = x_test)
            # This architecture is very similar to the
            # one proposed by the researchers
            print("[INFO] Creating new model")
            model = Sequential([
                Input(shape=x_train[0].shape),
                Conv2D(16, kernel_size=(2, 2), activation='relu'),
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            print("[INFO] Compiling model.")
            model.compile(loss = 'binary_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])
            print("[INFO] Model traning.")
            history = model.fit(x_train,
                y_train,
                verbose = 1,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks = [EarlyStopping(monitor = 'val_loss',
                                            patience = 5,
                                            verbose = 1)])
            print("[INFO] Final test accuracy: %.2f" %(model.evaluate(x_test, y_test)[1]))
            print("[INFO] Training completed.")
            # joint_model = Model(inputs = base_model.input, outputs = model.output)
            # plot_model(joint_model)
            return history, model
        else:
            print("[INFO] Creating new model with sliced VGG as non-trainable layer: saves memory but much slower.")
            model = Sequential([
                Input(shape=base_model.layers[-1].output_shape[1:]),
                Conv2D(16, kernel_size=(2, 2), activation='relu'),
                MaxPool2D(pool_size=(2, 2)),
                Flatten(),
                Dropout(0.3),
                Dense(50, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            print("[INFO] Joining models")
            for layer in base_model.layers:
                layer.trainable = False
            output = model(base_model.outputs)
            joined_models = Model(base_model.inputs, output)
            print("[INFO] Compiling model.")
            joined_models.compile(loss = 'binary_crossentropy',
                optimizer = optimizer,
                metrics = ['accuracy'])
            print("[INFO] Model traning.")
            history = joined_models.fit(preprocess_input(x_train),
                y_train,
                verbose = 1,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(preprocess_input(x_test), y_test))
            print("[INFO] Final test accuracy: %.2f" %(joined_models.evaluate(x_test, y_test)[1]))
            print("[INFO] Training completed.")
            # joint_model = Model(inputs = base_model.input, outputs = model.output)
            # plot_model(joint_model)
            return history, joined_models
    else:
        raise AttributeError("The channel dimension must be either 1 or 3. Others have not been implemented.")


def main():
    # load configurations
    print("[INFO] Loading vars")
    with open("./config.yml", 'r') as file:
        config_var = safe_load(file)["classifier"]
    data_paths = config_var['data_paths']
    print("[INFO] Loading data from hdf5 file.")
    #     in the final run, it should be called either 'positive' or 'negative'
    #     depending on which of the two classes I select
    train_ar = np.vstack((np.array(h5py.File(data_paths[0])['test']),
                          np.array(h5py.File(data_paths[1])['test'])))
    print("[INFO] RAM usage for data array: %.2fGB" % (float(getsizeof(train_ar)) * 1e-9))
    class_ar = np.append(
        np.zeros(h5py.File(data_paths[0])['test'].shape[0]) + 1,
        np.zeros(h5py.File(data_paths[1])['test'].shape[0]))
    print("[INFO] Preparing train-test split.")
    x_train, x_test, y_train, y_test = train_test_split(train_ar[:,config_var['vars_to_train'],:,:],
                                                    class_ar,
                                                    shuffle=True,
                                                    test_size=0.3)
    del train_ar
    history, model = transfer_learning_VGG(x_train = x_train, y_train = y_train, x_test = x_test,
                        y_test = y_test, slicing = config_var['slicing'], memory_efficient = config_var['memory_efficient'],
                        optimizer = 'adam', batch_size = 128, epochs = config_var['epochs'])
    print("[INFO] Saving model")
    model.save(config_var['output_name'])
    if config_var['make_plot'] is True:
        plt.figure(figsize = (10,10))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Train-test accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(config_var['image_output'])
        plt.show()
        plt.close()



if __name__ == "__main__":
    main()

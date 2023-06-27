import numpy as np
import matplotlib.pyplot as plt


def plot_loss_acc_curves(model_history):
    EPOCHS = len(model_history.history['loss'])
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(1,EPOCHS+1),model_history.history['loss'], )
    plt.plot(np.arange(1,EPOCHS+1),model_history.history['val_loss'])
    plt.xlabel("EPOCHS")
    plt.ylabel("LOSS")
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(np.arange(1,EPOCHS+1),model_history.history['accuracy'])
    plt.plot(np.arange(1,EPOCHS+1),model_history.history['val_accuracy'])
    plt.xlabel("EPOCHS")
    plt.ylabel("ACCURACY")
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title("Accuracy")

    plt.show();
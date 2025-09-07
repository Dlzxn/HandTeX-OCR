import torch
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.tensorboard import SummaryWriter



class Metrics:
    def __init__(self, num_classes, writer: SummaryWriter = None, class_names=None):
        """
        :param num_classes: количество классов
        :param writer: TensorBoard writer
        :param class_names: имена классов для матрицы ошибок
        """
        self.num_classes = num_classes
        self.writer = writer
        self.class_names = class_names if class_names else [str(i) for i in range(num_classes)]

    @staticmethod
    def compute_accuracy(y_true, y_pred):
        """
        Считаем простую точность
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	plt.figure(figsize=(10,5))
	plt.subplot(121)
	plt.plot(train_losses, label="Training Loss")
	plt.plot(valid_losses, label="Validation Loss")
	plt.xlabel("epoch")
	plt.ylabel("Loss")
	plt.title("Loss Curve")
	plt.legend()
	plt.subplot(122)
	plt.plot(train_accuracies, label="Training Accuracy")
	plt.plot(valid_accuracies, label="Validation Accuracy")
	plt.xlabel("epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy Curve")
	plt.legend()
	plt.savefig("learning_curves.png")
	plt.clf()
	
	


def plot_confusion_matrix(results, class_names):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	# used this example as base for code: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
	y_true, y_pred = zip(*results)
	y_true_label = [class_names[i] for i in y_true]
	y_pred_label = [class_names[i] for i in y_pred]
	cm = confusion_matrix(y_true_label, y_pred_label, labels = class_names)
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Normalized Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)
	cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).round(2)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig("confusion_matrix.png")
	plt.clf()

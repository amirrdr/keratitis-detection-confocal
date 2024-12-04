import glob
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import keras
from keras import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import *
import cv2
import os

data_dir = './Data/'
shape = (384, 384, 3)
EP = 150
BS = 30
CW = {0: 1, 1: 5, 2:30}
NM = ['Fungal', 'Acanthamoeba', 'Nonspecific']

FG_dir = './Data/'
FG_names = ['Fungal']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.jpg')
    FG_data.append([np.array(Image.open(fname).convert('L')) for fname in filelist])
FG_data = FG_data[0]
FG_data = np.array(FG_data)
FG_out = np.zeros((len(FG_data)))

FA_dir = './Data/Acanthamoeba/'
FA_names = ['Bright Spot', 'Cyst']
FA_data = []
for i in range(len(FA_names)):
    filelist = glob.glob(FA_dir + FA_names[i] + '/*.jpg')
    FA_data.append([np.array(Image.open(fname).convert('L')) for fname in filelist])
FA_data = np.concatenate((FA_data[0], FA_data[1]), axis=0)
FA_out = np.ones((len(FA_data)))

FNS_dir = './Data/'
FNS_names = ['Dendritic Cell', 'Nerve']
FNS_data = []
for i in range(len(FNS_names)):
    filelist = glob.glob(FNS_dir + FNS_names[i] + '/*.jpg')
    FNS_data.append([np.array(Image.open(fname).convert('L')) for fname in filelist])
FNS_data = np.concatenate((FNS_data[0], FNS_data[1]), axis=0)
FNS_out = 2 * np.ones((len(FNS_data)))

print('Shape of Fungal Keratitis Samples:', FG_data.shape)
print('Shape of Acanthamoeba Keratitis Samples:', FA_data.shape)
print('Shape of Nonspecific Keratitis Samples:', FNS_data.shape)

X = np.concatenate((FA_data[:, :384, :384], FG_data[:, :384, :384], FNS_data[:, :384, :384]), axis=0)
X_data = np.expand_dims(X, axis=3)
X_data = np.concatenate([X_data, X_data, X_data], axis=3)
y_ = np.concatenate((FA_out, FG_out, FNS_out), axis=0)
y_data = tf.keras.utils.to_categorical(y_, num_classes=3)

TS_dir = './Data/Harward/'
TS_names = ['Training Dataset', 'Test Dataset']
TS_data = []
for i in range(len(TS_names)):
    filelist = glob.glob(TS_dir + TS_names[i] + '/*.png')
    TS_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
TS_data = np.concatenate((TS_data[0], TS_data[1]), axis=0)
TS_data = np.expand_dims(TS_data, axis=3)
TS_data = np.concatenate([TS_data, TS_data, TS_data], axis=3)

TS_out = np.ones((len(TS_data)))
TS_out = tf.keras.utils.to_categorical(TS_out, num_classes=3)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
results = []

for train_index, val_index in kf.split(X_data):

    X_train, X_test = X_data[train_index], X_data[val_index]
    y_train, y_test = y_data[train_index], y_data[val_index]

    inp = keras.Input(shape=shape)

    x_1 = keras.layers.Conv2D(10, 3, padding='same', activation='relu')(inp)
    x = keras.layers.Dropout(0.1)(x_1)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(10, 15, activation='relu', strides=2)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(10, 30, activation='relu', strides=2)(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(15, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(5, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    out = keras.layers.Dense(3, activation='softmax')(x)

    model = Model(inp, out)
    heatmap_model = Model(inp, x_1)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    model.fit(X_train, y_train, validation_split=0.1, epochs=EP, batch_size=BS, shuffle=True, class_weight=CW)
    model.save('./Models/1.4/{}/Main Model/model.h5'.format(fold_no))
    heatmap_model.save('./Models/1.4/{}/Heatmap Model/model.h5'.format(fold_no))
    results = model.evaluate(X_test, y_test)
    print('Evaluation Results:', results)

    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_arg = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_arg, y_pred)
    precision = precision_score(y_test_arg, y_pred, average='weighted')
    recall = recall_score(y_test_arg, y_pred, average='weighted')
    f1 = f1_score(y_test_arg, y_pred, average='weighted')
    r2 = r2_score(y_test_arg, y_pred)
    report = classification_report(y_test_arg, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test_arg, y_pred)

    # Save classification report
    with open('./Results/1.4/{}/classification_report.txt'.format(fold_no), 'w') as f:
        f.write(classification_report(y_test_arg, y_pred))

    # Plot and save confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_arg, y_pred, normalize='true'), display_labels=NM)
    disp.plot()
    plt.savefig('./Results/1.4/{}/confusion_matrix.png'.format(fold_no), bbox_inches='tight')

    # Plot and save ROC curve
    plt.figure()
    fpr = dict()
    tpr = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
        plt.plot(fpr[i], tpr[i], lw=2, label='Class {}: {}'.format(NM[i], roc_auc_score(y_test[:, i], y_pred_proba[:, i])))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./Results/1.4/{}/roc_curve.png'.format(fold_no), bbox_inches='tight')

    # Plot and save Precision-Recall curve
    plt.figure()
    precision_plot = dict()
    recall_plot = dict()
    for i in range(3):
        precision_plot[i], recall_plot[i], _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
        plt.plot(recall_plot[i], precision_plot[i], lw=2, label='Class {}: {}'.format(NM[i], auc(recall_plot[i], precision_plot[i])))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('./Results/1.4/{}/precision_recall_curve.png'.format(fold_no), bbox_inches='tight')

    # Output all metrics
    print("Classification Report:")
    print(classification_report(y_test_arg, y_pred))

    print("Confusion Matrix:")
    print(conf_matrix)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("R2 Score:", r2)

    # Update prediction section
    output_dir_0 = './Results/1.4/{}/Heatmap/0/'.format(fold_no)
    output_dir_1 = './Results/1.4/{}/Heatmap/1/'.format(fold_no)
    output_dir_2 = './Results/1.4/{}/Heatmap/2/'.format(fold_no)

    os.makedirs(output_dir_0, exist_ok=True)
    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)

    def classify_and_save_with_gradcam(img, label, id):
        img_array = tf.expand_dims(img, axis=0)
        
        # Save original image and Grad-CAM heatmap based on predicted label
        if label == 0:
            img_save_path = os.path.join(output_dir_0, '{}_org.png'.format(id))
            cam_save_path = os.path.join(output_dir_0, '{}_cam.png'.format(id))
            sup_save_path = os.path.join(output_dir_0, '{}_sup.png'.format(id))
        elif label == 1:
            img_save_path = os.path.join(output_dir_1, '{}_org.png'.format(id))
            cam_save_path = os.path.join(output_dir_1, '{}_cam.png'.format(id))
            sup_save_path = os.path.join(output_dir_1, '{}_sup.png'.format(id))
        else:
            img_save_path = os.path.join(output_dir_2, '{}_org.png'.format(id))
            cam_save_path = os.path.join(output_dir_2, '{}_cam.png'.format(id))
            sup_save_path = os.path.join(output_dir_2, '{}_sup.png'.format(id))

        def make_gradcam_heatmap(img_array):
            grad_model = Model(
                inputs=[model.inputs],
                outputs=[heatmap_model.output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                predicted_class = tf.argmax(predictions[0])
                loss = predictions[:, predicted_class]

            grads = tape.gradient(loss, conv_outputs)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()

        # Generate Grad-CAM heatmap and save
        heatmap = make_gradcam_heatmap(img_array)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.4, heatmap, 1 - 0.4, 0)
        # Save images
        cv2.imwrite(img_save_path, img)
        cv2.imwrite(sup_save_path, superimposed_img)
        cv2.imwrite(cam_save_path, heatmap)

    for i in range(len(X_test)):
        classify_and_save_with_gradcam(X_test[i], y_test_arg[i], i)

    results = model.evaluate(TS_data, TS_out)
    print('External Validation Results:', results)

    fold_no += 1

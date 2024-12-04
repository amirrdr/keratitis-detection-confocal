import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import *

data_dir = './Data/'
shape = (384, 384, 1)
TS = 0.2
EP = 3
BS = 30
CW = {0: 1.2, 1: 0.6, 2:65}
NM = ['Fungal', 'Acanthamoeba', 'Nonspecific']

FG_dir = './Data/'
FG_names = ['Fungal']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.jpg')
    FG_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
FG_data = FG_data[0]
FG_data = np.array(FG_data)
FG_out = np.zeros((len(FG_data)))

FA_dir = './Data/Acanthamoeba/'
FA_names = ['Bright Spot', 'Cyst']
FA_data = []
for i in range(len(FA_names)):
    filelist = glob.glob(FA_dir + FA_names[i] + '/*.jpg')
    FA_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
FA_data = np.concatenate((FA_data[0], FA_data[1]), axis=0)
FA_out = np.ones((len(FA_data)))

FNS_dir = './Data/'
FNS_names = ['Dendritic Cell', 'Nerve']
FNS_data = []
for i in range(len(FNS_names)):
    filelist = glob.glob(FNS_dir + FNS_names[i] + '/*.jpg')
    FNS_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
FNS_data = np.concatenate((FNS_data[0], FNS_data[1]), axis=0)
FNS_out = 2 * np.ones((len(FNS_data)))

print('Shape of Fungal Keratitis Samples:', FG_data.shape)
print('Shape of Acanthamoeba Keratitis Samples:', FA_data.shape)
print('Shape of Nonspecific Keratitis Samples:', FNS_data.shape)

X = np.concatenate((FA_data[:, :384, :384], FG_data[:, :384, :384], FNS_data[:, :384, :384]), axis=0)
X = np.expand_dims(X, axis=3)
y_ = np.concatenate((FA_out, FG_out, FNS_out), axis=0)
y = tf.keras.utils.to_categorical(y_, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

inp = tf.keras.Input(shape=shape)

x_1 = tf.keras.layers.Conv2D(10, 10, strides=2, activation='relu')(inp)
x = tf.keras.layers.Dropout(0.1)(x_1)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(20, 10, strides=2, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(30, 10, strides=2, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.GlobalAvgPool2D()(x)
x = tf.keras.layers.Dense(15, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(5, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
out = tf.keras.layers.Dense(3, activation='softmax')(x)

model = Model(inp, out)
heatmap_model = Model(inp, x_1)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=TS / 4, epochs=EP, batch_size=BS, shuffle=True)
model.save('./Models/Main Model V1.2/model.h5')
heatmap_model.save('./Models/Heatmap Model V1.2/model.h5')
results = model.evaluate(X_test, y_test)
print('Evaluation Results:', results)
preds = model.predict(X_test)
preds = np.argmax(preds, axis=1)
y_test = np.argmax(y_test, axis=1)
print('Confusion Matrix:')
print(confusion_matrix(y_test, preds, normalize='true'))
print('Percision:')
print(precision_score(y_test, preds, average='weighted'))
print(precision_score(y_test, preds, average=None))
print('Recall (Sensitivity):')
print(recall_score(y_test, preds, average='weighted'))
print(recall_score(y_test, preds, average=None))
print('Specificity:')
print(recall_score(np.logical_not(y_test) , np.logical_not(preds), average='weighted'))
print(recall_score(np.logical_not(y_test) , np.logical_not(preds), average=None))
print('F1 Score:')
print(f1_score(y_test, preds, average='weighted'))
print(f1_score(y_test, preds, average=None))
print('R2 Score:')
print(r2_score(y_test, preds))

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, preds, normalize='true'), display_labels=NM)
disp.plot()
plt.show()
plt.savefig('./Results/Results V1.2/CM.JPG')

for i in range(len(NM)):
    a_1 = y_test - i + 1
    a_2 = preds - i + 1
    a_1[a_1 != 1] = 0
    a_2[a_2 != 1] = 0
    fpr, tpr, thresholds = metrics.roc_curve(a_1, a_2)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.savefig('./Results/Results V1.2/ROC (' + NM[i] + ' Vs. The Rest).JPG')

    prec, recall, _ = precision_recall_curve(a_1, a_2)
    print('PR AUC Value:', auc(recall, prec))
    display = PrecisionRecallDisplay.from_predictions(a_1, a_2)
    display.plot()
    plt.savefig('./Results/Results V1.2/PR (' + NM[i] + ' Vs. The Rest).JPG')

heats = heatmap_model.predict(X_test)
for i in range(len(heats)):
    tmp = 255 * (heats[i] - np.min(heats[i])) / np.ptp(heats[i])
    tmp = tmp[:, :, 0]
    result = Image.fromarray(tmp.astype(np.uint8))
    result.save('./Results/Results V1.2/Heatmaps/{}/'.format(y_test[i]) + str(i) + '_heatmap.png')
    tmp = 255 * (X_test[i] - np.min(X_test[i])) / np.ptp(X_test[i])
    tmp = tmp[:, :, 0]
    result = Image.fromarray(tmp.astype(np.uint8))
    result.save('./Results/Results V1.2/Heatmaps/{}/'.format(y_test[i]) + str(i) + '_original.png')

TS_dir = './Data/Harward/'
TS_names = ['Training Dataset', 'Test Dataset']
TS_data = []
for i in range(len(TS_names)):
    filelist = glob.glob(TS_dir + TS_names[i] + '/*.png')
    TS_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
TS_data = np.concatenate((TS_data[0], TS_data[1]), axis=0)
TS_data = np.expand_dims(TS_data, axis=3)

TS_out = np.ones((len(TS_data)))

results = model.evaluate(TS_data, TS_out)
print('External Validation Results:', results)

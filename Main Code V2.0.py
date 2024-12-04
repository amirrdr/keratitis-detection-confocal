import glob
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras import Model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import *

shape = (384, 384, 1)
EP = 100
BS = 10
CW = {0: 1, 1: 1.5}
NM = ['NSK', 'Acanthamoeba']

dir = './Data/Harward/'
names = ['Train Info MS.xlsx', 'Test Info MS (New).xlsx']

TR_imdir = 'Training Dataset'
TR_data = []
filelist = glob.glob(dir + TR_imdir + '/*.png')
TR_names = [fname.removesuffix('.png') for fname in filelist]
TR_names = [fname.removeprefix(dir + TR_imdir + '/') for fname in TR_names]
TR_names_int = [int(fname) for fname in TR_names]
TR_names_int = np.array(TR_names_int)

TR_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
TR_data = np.array(TR_data[0])

idx = pd.read_excel(dir + names[0], header=None, usecols="A")
idx = np.array(idx)
idx_ = []
for i in range(len(idx)-1):
    if idx[i] != 0:
        idx_.append(idx[i])
    else:
        tmp = idx[i - 1]
        while tmp != idx[i + 1]:
            tmp = tmp + 1
            idx_.append(tmp)
idx = idx_
idx = np.array(idx)
idx = idx[:, 0]
TR_data_0 = []
TR_data_1 = []
for i in range(len(TR_data)):
    if TR_names_int[i] in idx:
        TR_data_1.append(TR_data[i])
    else:
        TR_data_0.append(TR_data[i])

TS_imdir = 'Test Dataset'
TS_data = []
filelist = glob.glob(dir + TS_imdir + '/*.png')
TS_names = [fname.removesuffix('.png') for fname in filelist]
TS_names = [fname.removeprefix(dir + TS_imdir + '/') for fname in TS_names]
TS_names_int = [int(fname) for fname in TS_names]
TS_names_int = np.array(TS_names_int)

TS_data.append([np.array(Image.open(fname).resize((shape[1], shape[0])).convert('L')) for fname in filelist])
TS_data = np.array(TS_data[0])

idx = pd.read_excel(dir + names[1], header=None, usecols="A")
idx = np.array(idx)
idx_ = []
for i in range(len(idx)-1):
    if idx[i] != 0:
        idx_.append(idx[i])
    else:
        tmp = idx[i - 1]
        while tmp != idx[i + 1]:
            tmp = tmp + 1
            idx_.append(tmp)
idx = idx_
idx = np.array(idx)
idx = idx[:, 0]
TS_data_0 = []
TS_data_1 = []
for i in range(len(TS_data)):
    if TS_names_int[i] in idx:
        TS_data_1.append(TS_data[i])
    else:
        TS_data_0.append(TS_data[i])

X_train = np.concatenate((TR_data_0, TR_data_1), axis=0)
X_test = np.concatenate((TS_data_0, TS_data_1), axis=0)
print('Number of Samples in Training Dataset:', len(TR_data))
print('Number of Samples in Test Dataset:', len(TS_data))
y_train = np.concatenate((np.zeros(len(TR_data_0)), np.ones(len(TR_data_1))), axis=0)
y_test = np.concatenate((np.zeros(len(TS_data_0)), np.ones(len(TS_data_1))), axis=0)
print('Number of NSK Samples in Training Dataset:', len(TR_data_0))
print('Number of Acanthamoeba Samples in Training Dataset:', len(TR_data_1))
print('Number of NSK Samples in Test Dataset:', len(TS_data_0))
print('Number of Acanthamoeba Samples in Test Dataset:', len(TS_data_1))
X_train, y_train = shuffle(X_train, y_train, random_state=42)

inp = Input(shape=shape)

x_1 = Conv2D(10, 10, strides=2, activation='relu')(inp)
x = Dropout(0.1)(x_1)
x = BatchNormalization()(x)

x = Conv2D(20, 10, strides=2, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = Conv2D(30, 10, strides=2, activation='relu')(x)
x = Dropout(0.1)(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling2D()(x)
x = Dense(15, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(5, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inp, out)
heatmap_model = Model(inp, x_1)
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics='binary_accuracy')
model.summary()

model.fit(X_train, y_train, validation_split=0.1, epochs=EP, batch_size=BS, shuffle=True, class_weight=CW)
model.save('./Models/Main Model V2.0')
heatmap_model.save('./Models/Heatmap Model V2.0')
results = model.evaluate(X_test, y_test)
print('Evaluation Results:', results)
preds = model.predict(X_test)
preds[preds < 0.5] = 0
preds[preds >= 0.5] = 1
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

for i in range(len(NM)):
    a_1 = y_test - i + 1
    a_2 = preds - i + 1
    a_1[a_1 != 1] = 0
    a_2[a_2 != 1] = 0
    fpr, tpr, thresholds = metrics.roc_curve(a_1, a_2)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.savefig('./Results/Results V2.0/ROC (' + NM[i] + ' Vs. The Rest).JPG')

    prec, recall, _ = precision_recall_curve(a_1, a_2)
    print('PR AUC Value:', auc(recall, prec))
    display = PrecisionRecallDisplay.from_predictions(a_1, a_2)
    display.plot()
    plt.savefig('./Results/Results V2.0/PR (' + NM[i] + ' Vs. The Rest).JPG')

heats = heatmap_model.predict(X_test)
for i in range(len(heats)):
    tmp = 255 * (heats[i] - np.min(heats[i])) / np.ptp(heats[i])
    tmp = tmp[:, :, 0]
    result = Image.fromarray(tmp.astype(np.uint8))
    result.save('./Results/Results V2.0/Images/Heatmaps/' + str(i) + '.png')

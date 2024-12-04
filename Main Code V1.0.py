import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import *
from keras import Model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import *

data_dir = './Data/'
shape = (384, 384, 1)
TS = 0.2
EP = 150
BS = 30
CW = {0: 1, 1: 1.25, 2:10}
NM = ['Fungal', 'Acanthamoeba', 'Nonspecific']

FG_dir = './Data/'
FG_names = ['Fungal']
FG_data = []
for i in range(len(FG_names)):
    filelist = glob.glob(FG_dir + FG_names[i] + '/*.jpg')
    FG_data.append([np.array(Image.open(fname)) for fname in filelist])
FG_data = np.array(FG_data)
FG_data = FG_data[0]
FG_out = np.zeros((len(FG_data)))

FA_dir = './Data/Acanthamoeba/'
FA_names = ['Bright Spot', 'Cyst']
FA_data = []
for i in range(len(FA_names)):
    filelist = glob.glob(FA_dir + FA_names[i] + '/*.jpg')
    FA_data.append([np.array(Image.open(fname)) for fname in filelist])
FA_data = np.array(FA_data)
FA_data = np.concatenate((FA_data[0], FA_data[1]), axis=0)
FA_out = np.ones((len(FA_data)))

FNS_dir = './Data/'
FNS_names = ['Dendritic Cell', 'Nerve']
FNS_data = []
for i in range(len(FNS_names)):
    filelist = glob.glob(FNS_dir + FNS_names[i] + '/*.jpg')
    FNS_data.append([np.array(Image.open(fname)) for fname in filelist])
FNS_data = np.array(FNS_data)
FNS_data = np.concatenate((FNS_data[0], FNS_data[1]), axis=0)
FNS_out = 2 * np.ones((len(FNS_data)))

print('Number of Fungal Keratitis Samples:', len(FG_data))
print('Number of Acanthamoeba Keratitis Samples:', len(FA_data))
print('Number of Nonspecific Keratitis Samples:', len(FNS_data))

X = np.concatenate((FA_data[:, :384, :384], FG_data[:, :384, :384], FNS_data[:, :384, :384]), axis=0)
X = np.expand_dims(X, axis=3)
y_ = np.concatenate((FA_out, FG_out, FNS_out), axis=0)
y = tf.keras.utils.to_categorical(y_, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TS, shuffle=True)

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

x = GlobalAvgPool2D()(x)
x = Dense(15, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(5, activation='relu')(x)
x = Dropout(0.1)(x)
out = Dense(3, activation='softmax')(x)

model = Model(inp, out)
heatmap_model = Model(inp, x_1)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='categorical_accuracy')
model.summary()

model.fit(X_train, y_train, validation_split=TS / 4, epochs=EP, batch_size=BS, shuffle=True)
model.save('./Models/Main Model V1.0')
heatmap_model.save('./Models/Heatmap Model V1.0')
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

for i in range(len(NM)):
    a_1 = y_test - i + 1
    a_2 = preds - i + 1
    a_1[a_1 != 1] = 0
    a_2[a_2 != 1] = 0
    fpr, tpr, thresholds = metrics.roc_curve(a_1, a_2)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.savefig('./Results/Results V1.0/ROC (' + NM[i] + ' Vs. The Rest).JPG')

    prec, recall, _ = precision_recall_curve(a_1, a_2)
    print('PR AUC Value:', auc(recall, prec))
    display = PrecisionRecallDisplay.from_predictions(a_1, a_2)
    display.plot()
    plt.savefig('./Results/Results V1.0/PR (' + NM[i] + ' Vs. The Rest).JPG')

from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import classification_report, confusion_matrix

from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import metrics


audio_path = 'chick_sound/audio'
metadata_path = 'chick_sound/metadata/chick_sound.csv'
metadata = pd.read_csv(metadata_path)
feature = 'mgl'
#加载特征
X = np.load(f"extracted_features/chick_feat/X-{feature}.npy")
Y = np.load(f"extracted_features/chick_feat/Y-{feature}.npy")
# print(X.shape)
# DIR_SPEC = 'extracted_features/bird_feat/spec'
# X, Y = metrics.load_spectrogram_dataset(DIR_SPEC)
# flag = 'mgcc'
label_encoder = LabelEncoder()  #one-hot编码
Y_encoded = to_categorical(label_encoder.fit_transform(Y))  #这里要用fit_transform
num_fold = 1
num_itera = 1
model_name = "DenseNet121"

test_index, train_index = get_fold(num_fold)
X_test = np.take(X, test_index, axis=0)
Y_test_encoded = np.take(Y_encoded, test_index, axis=0)
X_train = np.take(X, train_index, axis=0)
Y_train_encoded = np.take(Y_encoded, train_index, axis=0)


if model_name=="DenseNet121":
    #加载mgcs模型
    custom_objects = {'SpatialAttention': SpatialAttention}
    loaded_model = models.load_model(f'E:/save_model/{model_name}/{feature}/fold{num_fold}_{num_itera}_model.h5',custom_objects=custom_objects)
    loaded_model.summary()
    #模型预测
    train_features = loaded_model.predict(X_train)
    test_features = loaded_model.predict(X_test)
    Y_train = np.argmax(Y_train_encoded, axis=1)
    # print(Y_train)
    svm_model = SVC(kernel='linear', C=40, gamma=0.0001)
    svm_model.fit(train_features, Y_train)
            # 在测试集上进行预测
    y_pred = svm_model.predict(test_features)      #获得预测值
            # svm计算准确率
    train = svm_model.score(train_features, Y_train)
    test = svm_model.score(test_features, np.argmax(Y_test_encoded, axis=1))
    unique_labels, label_counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique_labels, label_counts):
        print(f"Class {label}: {count} samples")

else:
# 对比模型
    loaded_model = models.load_model(f'E:/save_model/{model_name}/{feature}/fold{num_fold}_{num_itera}_model.h5')
    loaded_model.summary()
    # 模型预测
    train,test = metrics.evaluate_model(loaded_model, X_train, Y_train_encoded, X_test, Y_test_encoded)
    # train_results.append(train[1]*100)
    # test_results.append(test[1]*100)
    y_probs = loaded_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)  # 获得预测值


y_trues = np.argmax(Y_test_encoded, axis=1)
incorrect_indices = np.where(y_pred != y_trues)[0]
# 打印分类错误的样本
for idx in incorrect_indices:
    print(f"样本索引: {idx},在csv文件中的位置:{test_index[idx]},类别为{metadata.iloc[test_index[idx],0]} 真实标签: {y_trues[idx]}, 预测标签: {y_pred[idx]}")

# CM = confusion_matrix(y_trues, y_pred)
labels =['cough', 'crow', 'feeder', 'flap', 'normal', 'peck', 'scream','snore','ventilator']
re = classification_report(y_trues, y_pred, labels=[0,1,2,3,4,5,6,7,8], target_names=labels,digits=5)
# metrics.display_cm(y_trues,y_pred,num_fold,num_itera,model_name,feature)
print(re)

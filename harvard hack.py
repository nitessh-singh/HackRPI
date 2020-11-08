{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\ops\\nn_impl.py:180: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use tf.where in 2.0, which has the same broadcast rule as np.where\n",
      "Train on 20307 samples, validate on 2257 samples\n",
      "Epoch 1/5\n",
      "20307/20307 [==============================] - 493s 24ms/sample - loss: 0.5920 - acc: 0.6817 - val_loss: 0.5201 - val_acc: 0.7559\n",
      "Epoch 2/5\n",
      "20307/20307 [==============================] - 468s 23ms/sample - loss: 0.5136 - acc: 0.7526 - val_loss: 0.5007 - val_acc: 0.7740\n",
      "Epoch 3/5\n",
      "20307/20307 [==============================] - 460s 23ms/sample - loss: 0.4716 - acc: 0.7795 - val_loss: 0.4847 - val_acc: 0.7802\n",
      "Epoch 4/5\n",
      "20307/20307 [==============================] - 477s 23ms/sample - loss: 0.4180 - acc: 0.8103 - val_loss: 0.4723 - val_acc: 0.7851\n",
      "Epoch 5/5\n",
      "20307/20307 [==============================] - 454s 22ms/sample - loss: 0.3596 - acc: 0.8405 - val_loss: 0.5298 - val_acc: 0.7612\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization\n",
    "from tensorflow.keras.callbacks import TensorBoard\n",
    "import os\n",
    "import pickle\n",
    "import time\n",
    "\n",
    "NAME = \"HARVARD-HACK-{}\".format(int(time.time()))\n",
    "\n",
    "tensorboard = TensorBoard(log_dir='harvard_hack_logs/{}'.format(NAME))\n",
    "tboard_log_dir = os.path.join(\"harvard_hack_logs\",NAME)\n",
    "tensorboard = TensorBoard(log_dir = tboard_log_dir)\n",
    "\n",
    "x = pickle.load(open(\"x.pickle\",\"rb\"))\n",
    "y = pickle.load(open(\"y.pickle\",\"rb\"))\n",
    "\n",
    "x = x/255\n",
    "\n",
    "model = Sequential()\n",
    "#model.add(Conv2D(64, (3,3), input_shape = x.shape[1:], activation='relu'))\n",
    "model.add(Conv2D(64, (3,3), input_shape = x.shape[1:], activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))   \n",
    "\n",
    "model.add(Conv2D(64, (3,3), activation='relu'))\n",
    "model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64, activation = 'relu'))\n",
    "\n",
    "model.add(Dense(1,activation = 'sigmoid'))\n",
    "          \n",
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])\n",
    "model.fit(x,y,batch_size=32,epochs=5,validation_split = 0.1, callbacks=[tensorboard])\n",
    "\n",
    "model.save('models/wasteclassifier', overwrite=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "20307/20307 [==============================] - 464s 23ms/sample - loss: 0.3294 - acc: 0.8622\n",
      "0.8626495347806823\n",
      "[[1122  135]\n",
      " [ 175  825]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.87      0.89      0.88      1257\n",
      "           1       0.86      0.82      0.84      1000\n",
      "\n",
      "    accuracy                           0.86      2257\n",
      "   macro avg       0.86      0.86      0.86      2257\n",
      "weighted avg       0.86      0.86      0.86      2257\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10)\n",
    "model.fit(x_train, y_train)\n",
    "y_pred = model.predict_classes(x_test)\n",
    "print(accuracy_score(y_test,y_pred))\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x26f041fc4c8>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import sklearn.metrics as metrics\n",
    "cm= metrics.confusion_matrix(y_test,y_pred)\n",
    "sns.heatmap(cm, annot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'plot_model_result' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-2-dbb8bbf3b56f>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mplot_model_result\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'plot_model_result' is not defined"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\Lenovo\\anaconda3\\Scripts\\conda-script.py\", line 11, in <module>\n",
      "    from conda.cli import main\n",
      "ModuleNotFoundError: No module named 'conda'\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved to the disk.\n"
     ]
    }
   ],
   "source": [
    "# Save the Model Weights\n",
    "model.save_weights('model_100_eopchs_adam_20191030_01.h5')\n",
    "\n",
    "# Save the Model to JSON\n",
    "model_json = model.to_json()\n",
    "with open('model_adam_20191030_01.json', 'w') as json_file:\n",
    "    json_file.write(model_json)\n",
    "    \n",
    "print('Model saved to the disk.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\dtypes.py:525: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:541: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint8 = np.dtype([(\"qint8\", np.int8, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:542: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint8 = np.dtype([(\"quint8\", np.uint8, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:543: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint16 = np.dtype([(\"qint16\", np.int16, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:544: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_quint16 = np.dtype([(\"quint16\", np.uint16, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:545: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  _np_qint32 = np.dtype([(\"qint32\", np.int32, 1)])\n",
      "C:\\Users\\Lenovo\\anaconda3\\lib\\site-packages\\tensorboard\\compat\\tensorflow_stub\\dtypes.py:550: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.\n",
      "  np_resource = np.dtype([(\"resource\", np.ubyte, 1)])\n"
     ]
    },
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'vb100_utils'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-888fc9f51a9b>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     15\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mPIL\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mImage\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     16\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mPIL\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 17\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mvb100_utils\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[1;33m*\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'vb100_utils'"
     ]
    }
   ],
   "source": [
    "# ------------------------------------------------------------------------\n",
    "# Load saved model and its weights\n",
    "'''\n",
    ">> Model weights are saved to HDF5 format.\n",
    ">> The model structure can be described and saved using two different formats: JSON and YAML.\n",
    "'''\n",
    "\n",
    "# Import dependencies\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.models import model_from_json\n",
    "from tensorflow.python.framework import ops\n",
    "ops.reset_default_graph()\n",
    "import h5py \n",
    "#import itertools\n",
    "from PIL import Image\n",
    "import PIL\n",
    "from vb100_utils import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print('h5py version is {}'.format(h5py.__version__))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get the architecture of CNN\n",
    "json_file = open('./good_models/20190807/02/model_adam.json')\n",
    "loaded_model_json = json_file.read()\n",
    "json_file.close()\n",
    "loaded_model = model_from_json(loaded_model_json)\n",
    "\n",
    "# Get weights into the model\n",
    "loaded_model.load_weights('./good_models/20190807/02/model_100_eopchs_adam_20190807.h5')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

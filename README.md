{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <center>Malria Disease Prediction</center>\n",
    "\n",
    "## Overview:\n",
    "Malaria is a life-threatening disease caused by parasites transmitted through the bite of infected female Anopheles mosquitoes. Symptoms typically include fever, chills, and flu-like illness, and if left untreated, it can lead to severe complications and death, particularly in children under five and pregnant women. Prevention strategies include the use of insecticide-treated bed nets, indoor residual spraying, and antimalarial medications. Malaria remains a significant public health challenge, particularly in tropical and subtropical regions of the world.\n",
    "\n",
    "* We are trying to detect whether the cell/tissue is Infected or not.\n",
    "## Dataset Information:\n",
    "\n",
    "Malaria Infected Tissues Dataset - [Kaggle - Malaria Disease Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data)\n",
    "\n",
    "The Malaria dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells from the thin blood smear slide images of segmented cells.\n",
    "\n",
    "### Training : Testing ::   22048 : 5510"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Importing required libraries\n",
    "* ```Numpy```: A fundamental package for scientific computing in Python. It provides support for multidimensional arrays, along with a collection of mathematical functions to operate on these arrays efficiently. NumPy is widely used in numerical and scientific computing tasks, including data manipulation, linear algebra, statistics, and signal processing.\n",
    "\n",
    "* ```Pandas```: A powerful library for data manipulation and analysis in Python. It offers data structures and functions for working with structured data, primarily in the form of dataframes. Dataframes are two-dimensional labeled arrays capable of holding heterogeneous data types. Pandas provides tools for reading and writing data from various file formats, reshaping and transforming data, and performing data analysis tasks such as grouping, filtering, and aggregation.\n",
    "\n",
    "* ```Matplotlib```: A plotting library for creating visualizations in Python. It provides a MATLAB-like interface for generating a wide range of static, interactive, and animated plots. Matplotlib is highly customizable and supports various plot types, including line plots, scatter plots, bar charts, histograms, and heatmaps.\n",
    "\n",
    "* ```TensorFlow```: TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and resources for building and deploying machine learning models at scale."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:45:37.431517Z",
     "iopub.status.busy": "2024-02-21T15:45:37.431145Z",
     "iopub.status.idle": "2024-02-21T15:45:41.985153Z",
     "shell.execute_reply": "2024-02-21T15:45:41.984431Z",
     "shell.execute_reply.started": "2024-02-21T15:45:37.431483Z"
    },
    "id": "l9z2oNNdi80k"
   },
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.simplefilter('ignore')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import Sequential\n",
    "from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.callbacks import EarlyStopping"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:45:41.987329Z",
     "iopub.status.busy": "2024-02-21T15:45:41.987025Z",
     "iopub.status.idle": "2024-02-21T15:45:41.991872Z",
     "shell.execute_reply": "2024-02-21T15:45:41.990881Z",
     "shell.execute_reply.started": "2024-02-21T15:45:41.987298Z"
    },
    "id": "U69K2DVKjbWm"
   },
   "source": [
    "# 2. Data Preproceessing"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Displaying Uninfected and Infected Cell tissues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:45:41.993349Z",
     "iopub.status.busy": "2024-02-21T15:45:41.993086Z",
     "iopub.status.idle": "2024-02-21T15:45:42.437029Z",
     "shell.execute_reply": "2024-02-21T15:45:42.436015Z",
     "shell.execute_reply.started": "2024-02-21T15:45:41.993324Z"
    }
   },
   "outputs": [
    {
     "data": {

      "text/plain": [
       "<Figure size 1080x504 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import cv2\n",
    "upic='../input/cell-images-for-detecting-malaria/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_131.png'\n",
    "apic='../input/cell-images-for-detecting-malaria/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_164.png'\n",
    "plt.figure(1, figsize = (15 , 7))\n",
    "plt.subplot(1 , 2 , 1)\n",
    "plt.imshow(cv2.imread(upic))\n",
    "plt.title('Uninfected Cell')\n",
    "plt.xticks([]) , plt.yticks([])\n",
    "\n",
    "plt.subplot(1 , 2 , 2)\n",
    "plt.imshow(cv2.imread(apic))\n",
    "plt.title('Infected Cell')\n",
    "plt.xticks([]) , plt.yticks([])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.2 Training and Testing Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:45:42.446994Z",
     "iopub.status.busy": "2024-02-21T15:45:42.446581Z",
     "iopub.status.idle": "2024-02-21T15:45:42.451952Z",
     "shell.execute_reply": "2024-02-21T15:45:42.451188Z",
     "shell.execute_reply.started": "2024-02-21T15:45:42.446957Z"
    },
    "id": "MAd3M91Smb3j"
   },
   "outputs": [],
   "source": [
    "datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:45:42.453417Z",
     "iopub.status.busy": "2024-02-21T15:45:42.453147Z",
     "iopub.status.idle": "2024-02-21T15:46:18.557437Z",
     "shell.execute_reply": "2024-02-21T15:46:18.556415Z",
     "shell.execute_reply.started": "2024-02-21T15:45:42.453390Z"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 22048 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "trainDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',\n",
    "                                           target_size=(128,128),\n",
    "                                           class_mode = 'binary',\n",
    "                                           batch_size = 16,\n",
    "                                           subset='training')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:18.558959Z",
     "iopub.status.busy": "2024-02-21T15:46:18.558697Z",
     "iopub.status.idle": "2024-02-21T15:46:24.267183Z",
     "shell.execute_reply": "2024-02-21T15:46:24.266206Z",
     "shell.execute_reply.started": "2024-02-21T15:46:18.558933Z"
    },
    "id": "7I7zFilVoUDm",
    "outputId": "fe3e90da-2503-4e36-e206-6b8cacfffcba"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 5510 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "valDatagen = datagen.flow_from_directory(directory='../input/cell-images-for-detecting-malaria/cell_images/cell_images/',\n",
    "                                           target_size=(128,128),\n",
    "                                           class_mode = 'binary',\n",
    "                                           batch_size = 16,\n",
    "                                           subset='validation')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Modelling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3.1 Convolutional Neural Networks Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:24.268722Z",
     "iopub.status.busy": "2024-02-21T15:46:24.268430Z",
     "iopub.status.idle": "2024-02-21T15:46:27.204229Z",
     "shell.execute_reply": "2024-02-21T15:46:27.203477Z",
     "shell.execute_reply.started": "2024-02-21T15:46:24.268692Z"
    },
    "id": "eGj3RRNRomKa"
   },
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))\n",
    "model.add(MaxPool2D(2,2))\n",
    "model.add(Dropout(0.2))\n",
    "\n",
    "model.add(Conv2D(32,(3,3),activation='relu'))\n",
    "model.add(MaxPool2D(2,2))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Conv2D(64,(3,3),activation='relu'))\n",
    "model.add(MaxPool2D(2,2))\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Flatten())\n",
    "model.add(Dense(64,activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "\n",
    "model.add(Dense(1,activation='sigmoid'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:27.205757Z",
     "iopub.status.busy": "2024-02-21T15:46:27.205395Z",
     "iopub.status.idle": "2024-02-21T15:46:27.213468Z",
     "shell.execute_reply": "2024-02-21T15:46:27.212424Z",
     "shell.execute_reply.started": "2024-02-21T15:46:27.205724Z"
    },
    "id": "3qesYVQMpqn6",
    "outputId": "ab924654-3993-48ea-f843-ba09426c5243"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d (Conv2D)              (None, 126, 126, 16)      448       \n",
      "_________________________________________________________________\n",
      "max_pooling2d (MaxPooling2D) (None, 63, 63, 16)        0         \n",
      "_________________________________________________________________\n",
      "dropout (Dropout)            (None, 63, 63, 16)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_1 (Conv2D)            (None, 61, 61, 32)        4640      \n",
      "_________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2 (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "dropout_1 (Dropout)          (None, 30, 30, 32)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_2 (Conv2D)            (None, 28, 28, 64)        18496     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         \n",
      "_________________________________________________________________\n",
      "dropout_2 (Dropout)          (None, 14, 14, 64)        0         \n",
      "_________________________________________________________________\n",
      "flatten (Flatten)            (None, 12544)             0         \n",
      "_________________________________________________________________\n",
      "dense (Dense)                (None, 64)                802880    \n",
      "_________________________________________________________________\n",
      "dropout_3 (Dropout)          (None, 64)                0         \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 1)                 65        \n",
      "=================================================================\n",
      "Total params: 826,529\n",
      "Trainable params: 826,529\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:27.215913Z",
     "iopub.status.busy": "2024-02-21T15:46:27.215382Z",
     "iopub.status.idle": "2024-02-21T15:46:27.232619Z",
     "shell.execute_reply": "2024-02-21T15:46:27.231854Z",
     "shell.execute_reply.started": "2024-02-21T15:46:27.215870Z"
    },
    "id": "3TOyOrioptrq"
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:27.234093Z",
     "iopub.status.busy": "2024-02-21T15:46:27.233763Z",
     "iopub.status.idle": "2024-02-21T15:46:27.239799Z",
     "shell.execute_reply": "2024-02-21T15:46:27.238738Z",
     "shell.execute_reply.started": "2024-02-21T15:46:27.234058Z"
    }
   },
   "outputs": [],
   "source": [
    "early_stop = EarlyStopping(monitor='val_loss',patience=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:46:27.241277Z",
     "iopub.status.busy": "2024-02-21T15:46:27.240974Z",
     "iopub.status.idle": "2024-02-21T15:53:05.311456Z",
     "shell.execute_reply": "2024-02-21T15:53:05.310671Z",
     "shell.execute_reply.started": "2024-02-21T15:46:27.241250Z"
    },
    "id": "dSnFw-V2p2B9",
    "outputId": "e59014ac-3bf1-41c9-b7f7-5b2d84835949"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/20\n",
      "1378/1378 [==============================] - 188s 136ms/step - loss: 0.5599 - accuracy: 0.6957 - val_loss: 0.4062 - val_accuracy: 0.9312\n",
      "Epoch 2/20\n",
      "1378/1378 [==============================] - 51s 37ms/step - loss: 0.2111 - accuracy: 0.9354 - val_loss: 0.1643 - val_accuracy: 0.9396\n",
      "Epoch 3/20\n",
      "1378/1378 [==============================] - 51s 37ms/step - loss: 0.1731 - accuracy: 0.9477 - val_loss: 0.1620 - val_accuracy: 0.9354\n",
      "Epoch 4/20\n",
      "1378/1378 [==============================] - 52s 38ms/step - loss: 0.1612 - accuracy: 0.9524 - val_loss: 0.1726 - val_accuracy: 0.9385\n",
      "Epoch 5/20\n",
      "1378/1378 [==============================] - 52s 38ms/step - loss: 0.1481 - accuracy: 0.9548 - val_loss: 0.1874 - val_accuracy: 0.9399\n"
     ]
    }
   ],
   "source": [
    "history = model.fit_generator(generator = trainDatagen,\n",
    "                             steps_per_epoch = len(trainDatagen),\n",
    "                              epochs =20,\n",
    "                              validation_data = valDatagen,\n",
    "                              validation_steps=len(valDatagen),\n",
    "                             callbacks=[early_stop])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:53:05.313121Z",
     "iopub.status.busy": "2024-02-21T15:53:05.312842Z",
     "iopub.status.idle": "2024-02-21T15:53:05.322137Z",
     "shell.execute_reply": "2024-02-21T15:53:05.321223Z",
     "shell.execute_reply.started": "2024-02-21T15:53:05.313091Z"
    },
    "id": "qWX7Iy53qTZy"
   },
   "outputs": [],
   "source": [
    "def plotLearningCurve(history,epochs):\n",
    "  epochRange = range(1,epochs+1)\n",
    "  plt.plot(epochRange,history.history['accuracy'])\n",
    "  plt.plot(epochRange,history.history['val_accuracy'])\n",
    "  plt.title('Model Accuracy')\n",
    "  plt.xlabel('Epoch')\n",
    "  plt.ylabel('Accuracy')\n",
    "  plt.legend(['Train','Validation'],loc='upper left')\n",
    "  plt.show()\n",
    "\n",
    "  plt.plot(epochRange,history.history['loss'])\n",
    "  plt.plot(epochRange,history.history['val_loss'])\n",
    "  plt.title('Model Loss')\n",
    "  plt.xlabel('Epoch')\n",
    "  plt.ylabel('Loss')\n",
    "  plt.legend(['Train','Validation'],loc='upper left')\n",
    "  plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-02-21T15:58:38.580050Z",
     "iopub.status.busy": "2024-02-21T15:58:38.579667Z",
     "iopub.status.idle": "2024-02-21T15:58:38.906414Z",
     "shell.execute_reply": "2024-02-21T15:58:38.905654Z",
     "shell.execute_reply.started": "2024-02-21T15:58:38.580019Z"
    },
    "id": "jdB5PpS1sIhE",
    "outputId": "75956d00-c8c1-418a-dcf3-79895b908b05"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nO3deXxU9b3/8dcn+wYkIQhKIAluCEUgBBQFxK11q7hWaatSVKpWrfXWqr1tpe31dtG2XlvbXpfq1Wr5ee11qXVpoSq4solVBCxo0ICyJIQsZM/398eZhCFMQpY5M5PM+/l4zCMz55z5fj85hPnM+X7P+RxzziEiIvErIdoBiIhIdCkRiIjEOSUCEZE4p0QgIhLnlAhEROKcEoGISJxTIhA5ADMrNDNnZknd2Haemb0aibhEwkWJQAYUMys1s0Yzy+uwfE3gw7wwOpH1LKGIRJISgQxEHwFz216Y2QQgPXrhiMQ2JQIZiB4BLg16fRnwcPAGZjbEzB42sx1mttnMvmdmCYF1iWZ2p5ntNLMPgTNDvPcBM/vUzLaY2X+YWWJfAjazQ8zsGTOrMLONZnZl0LppZrbSzKrMbJuZ/TKwPM3M/mhm5WZWaWYrzGx4X+KQ+KREIAPRm8BgMzsq8AF9EfDHDtv8GhgCjAFOwEscXwusuxI4C5gMlAAXdHjv/wDNwGGBbT4PXNHHmP8ElAGHBPr7TzM7ObDuv4D/cs4NBg4FHg8svyzwO4wChgJXAXV9jEPikBKBDFRtRwWnAuuBLW0rgpLDrc65audcKfAL4JLAJl8C7nLOfeKcqwB+EvTe4cDpwA3OuVrn3HbgV8DFvQ3UzEYBM4CbnXP1zrk1wP1B8TQBh5lZnnOuxjn3ZtDyocBhzrkW59wq51xVb+OQ+KVEIAPVI8CXgXl0GBYC8oAUYHPQss3AyMDzQ4BPOqxrUwAkA58GhmMqgf8GDupDrIcAFc656k7iuRw4AlgfGP45K7D8EeBFYJGZbTWzn5tZch/ikDilRCADknNuM96k8RnA/3VYvRPv23RB0LLR7D1q+BRvuCV4XZtPgAYgzzmXHXgMds6N70O4W4FcMxsUKh7n3L+cc3Pxks3PgCfMLNM51+Sc+6FzbhxwHN5w1qWI9JASgQxklwMnOedqgxc651rwxtlvN7NBZlYA3MjeeYTHgevNLN/McoBbgt77KfA34BdmNtjMEszsUDM7oQdxpQYmetPMLA3vA/914CeBZUcHYn8UwMy+ambDnHOtQGWgjRYzO9HMJgSGuqrwkltLD+IQAZQIZABzzm1yzq3sZPV1QC3wIfAq8Bjwh8C6+/CGXN4BVrP/EcWleENL7wO7gCeAg3sQWg3epG7b4yS8010L8Y4OngRuc879PbD9acBaM6vBmzi+2DlXD4wI9F0FrANeYf9JcZEDMt2YRkQkvumIQEQkzikRiIjEOSUCEZE4p0QgIhLn+l0VxLy8PFdYWBjtMERE+pVVq1btdM4NC7Wu3yWCwsJCVq7s7IxAEREJxcw2d7ZOQ0MiInFOiUBEJM4pEYiIxLl+N0cQSlNTE2VlZdTX10c7lAEjLS2N/Px8kpNVzFJkoBsQiaCsrIxBgwZRWFiImUU7nH7POUd5eTllZWUUFRVFOxwR8dmAGBqqr69n6NChSgJhYmYMHTpUR1gicWJAJAJASSDMtD9F4seASQQH0tDUwtbKOlpVbVVEZB/xkwiaW9lZ08Duuqawt11eXs6kSZOYNGkSI0aMYOTIke2vGxsbu3zvypUruf7668Mek4hIdw2IyeLuGJSWRFpSIjurG8hOTw7r0MfQoUNZs2YNAAsXLiQrK4tvf/vb7eubm5tJSgq9q0tKSigpKQlbLCIiPRU3RwRmRt6gVOqaWqhpaPa9v3nz5nHjjTdy4okncvPNN7N8+XKOO+44Jk+ezHHHHceGDRsAePnllznrLO9e5AsXLmT+/PnMnj2bMWPGcPfdd/sep4jIgDsi+OFf1vL+1qpO1+9pbCHBIC05sdttjjtkMLd9sef3Jv/ggw9YvHgxiYmJVFVVsXTpUpKSkli8eDHf/e53+fOf/7zfe9avX89LL71EdXU1Rx55JFdffbXO5RcRXw24RHAgyYlGY3Mrrc6R4POZMRdeeCGJiV7C2b17N5dddhn/+te/MDOamkLPVZx55pmkpqaSmprKQQcdxLZt28jPz/c1ThGJbwMuERzom3tzSyvrP6tmSHoyo3IzfI0lMzOz/fn3v/99TjzxRJ588klKS0uZPXt2yPekpqa2P09MTKS52f9hLBGJb3EzR9AmKTGB3MwUKvc00djcGrF+d+/ezciRIwF46KGHItaviMiBxF0iAMjLSgEc5bUNEevzO9/5DrfeeivHH388LS0tEetXRORAzPWzC6xKSkpcxxvTrFu3jqOOOqpH7XxcXkt1fTNjDx5EYkJc5sMD6s1+FZHYZGarnHMhz1WP20/AvEGptDhHRW34LzATEelP4jYRZKQkkZmaxM6aBpWdEJG4FreJAGBYVipNLa2+lJ0QEekv4joRDEpLIjVQdqK/zZWIiIRLXCcCM2NYoOxEbQTKToiIxKK4TgQA2RnJJCUksKOm6yqhIiIDVdwnggQz8rJSqK5voq6xd+f3z549mxdffHGfZXfddRfXXHNNp9u3nQJ7xhlnUFlZud82Cxcu5M477+yy36eeeor333+//fUPfvADFi9e3NPwRSTOxX0iAMjNTCHBjJ01vbvAbO7cuSxatGifZYsWLWLu3LkHfO9zzz1HdnZ2r/rtmAh+9KMfccopp/SqLRGJX0oEBJWdqGuiqRdlJy644AKeffZZGhq8RFJaWsrWrVt57LHHKCkpYfz48dx2220h31tYWMjOnTsBuP322znyyCM55ZRT2stUA9x3331MnTqViRMncv7557Nnzx5ef/11nnnmGW666SYmTZrEpk2bmDdvHk888QQAS5YsYfLkyUyYMIH58+e3x1ZYWMhtt91GcXExEyZMYP369T3+fUVkYBlwRed4/hb47N0ev22EcwxubKE1ySCxQ4nqERPg9J92+t6hQ4cybdo0XnjhBebMmcOiRYu46KKLuPXWW8nNzaWlpYWTTz6Zf/7znxx99NEh21i1ahWLFi3i7bffprm5meLiYqZMmQLAeeedx5VXXgnA9773PR544AGuu+46zj77bM466ywuuOCCfdqqr69n3rx5LFmyhCOOOIJLL72U3/3ud9xwww0A5OXlsXr1an77299y5513cv/99/d4f4nIwKEjgoAEM5ISjeYWh6Pnp5IGDw+1DQs9/vjjFBcXM3nyZNauXbvPME5Hy5Yt49xzzyUjI4PBgwdz9tlnt6977733mDlzJhMmTODRRx9l7dq1XcayYcMGioqKOOKIIwC47LLLWLp0afv68847D4ApU6ZQWlra499VRAaWgXdE0MU39wNpbWxm0/YaDh6SzrBBqQd+Q5BzzjmHG2+8kdWrV1NXV0dOTg533nknK1asICcnh3nz5lFfX99lG53dPnPevHk89dRTTJw4kYceeoiXX365y3YOdE1EW6lrlbkWEdARwT76UnYiKyuL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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plotLearningCurve(history,5)"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "gpu",
   "dataSources": [
    {
     "datasetId": 87153,
     "sourceId": 200743,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 29943,
   "isGpuEnabled": true,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

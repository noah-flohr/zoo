from typing import Tuple
import tensorflow as tf
import numpy as np
import larq as lq
import math

import os
proj_path = os.path.abspath(os.path.dirname(__file__))
res_path = f"{proj_path}/results"
os.makedirs(res_path, exist_ok=True)

# TernaryDenseNet
from larq_zoo.literature.densenet_ternary import *
from larq_zoo.training.basic_experiments import TrainTernaryDenseNet28, TrainTernaryDenseNet37

# TerRealNet
from larq_zoo.literature.birealnet import *
from larq_zoo.training.basic_experiments import TrainTerRealNet

# ResNetE18
from larq_zoo.literature.resnet_e import *
from larq_zoo.training.basic_experiments import TrainTernaryResNetE18


def _get_dataset(dataset: str = "cifar10") -> Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return 10, (x_train, y_train), (x_test, y_test)
    elif dataset == "cifar100" :
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        return 100, (x_train, y_train), (x_test, y_test)
    else :
        raise Exception(f"Dataset {dataset} not supported.")


def _fit(nn, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, train_params) -> None:
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(train_params.batch_size).repeat()
    callbacks = []
    if hasattr(train_params, "learning_rate_schedule"):
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(train_params.learning_rate_schedule)
        )
    nn.fit(x_train, y_train,
        epochs=train_params.epochs,
        steps_per_epoch=math.ceil(x_train.shape[0] / train_params.batch_size),
        validation_data=(x_test, y_test),
        validation_steps=math.ceil(x_test.shape[0] / train_params.batch_size),
        validation_freq=train_params.validation_frequency,
        verbose=1,
        callbacks=callbacks
    )



def Mod_TrainTerRealNet(dataset: str = "cifar10") -> None :
    num_classes, (x_train, y_train), (x_test, y_test) = _get_dataset(dataset)

    train_params = TrainTerRealNet()

    nn = BiRealNet(
        input_shape=x_train.shape[1:],
        weights=None,
        num_classes=num_classes
    )

    metrics = ["sparse_categorical_accuracy"]
    loss = "sparse_categorical_crossentropy"
    nn.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=metrics,
    )
    lq.models.summary(nn)
    _fit(nn, x_train, y_train, x_test, y_test, train_params)
    nn.save(f"{res_path}/{dataset}_TerRealNet.h5")


def Mod_TrainTernaryResNetE18(dataset: str = "cifar10") -> None :
    num_classes, (x_train, y_train), (x_test, y_test) = _get_dataset(dataset)

    train_params = TrainTernaryResNetE18()

    nn = TernaryResNetE18(
        input_shape=x_train.shape[1:],
        weights=None,
        num_classes=num_classes
    )

    metrics = ["sparse_categorical_accuracy"]
    loss = "sparse_categorical_crossentropy"
    nn.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=metrics,
    )
    lq.models.summary(nn)
    _fit(nn, x_train, y_train, x_test, y_test, train_params)
    nn.save(f"{res_path}/{dataset}_TernaryResNetE18.h5")



def Mod_TrainTernaryDenseNet28(dataset: str = "cifar10") -> None :
    num_classes, (x_train, y_train), (x_test, y_test) = _get_dataset(dataset)

    train_params = TrainTernaryDenseNet28()

    nn = TernaryDenseNet28 (
        input_shape=x_train.shape[1:],
        weights=None,
        num_classes=num_classes
    )

    metrics = ["sparse_categorical_accuracy"]
    loss = "sparse_categorical_crossentropy"
    nn.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=metrics,
    )
    lq.models.summary(nn)
    _fit(nn, x_train, y_train, x_test, y_test, train_params)
    nn.save(f"{res_path}/{dataset}_TernaryDenseNet28.h5")


def Mod_TrainTernaryDenseNet37(dataset: str = "cifar10") -> None :
    num_classes, (x_train, y_train), (x_test, y_test) = _get_dataset(dataset)

    train_params = TrainTernaryDenseNet37()

    nn = TernaryDenseNet28 (
        input_shape=x_train.shape[1:],
        weights=None,
        num_classes=num_classes
    )

    metrics = ["sparse_categorical_accuracy"]
    loss = "sparse_categorical_crossentropy"
    nn.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=metrics,
    )
    lq.models.summary(nn)
    _fit(nn, x_train, y_train, x_test, y_test, train_params)
    nn.save(f"{res_path}/{dataset}_TernaryDenseNet37.h5")

if __name__ == "__main__":
    Mod_TrainTernaryDenseNet37("cifar10")
    Mod_TrainTernaryDenseNet37("cifar100")
    Mod_TrainTernaryResNetE18("cifar10")
    Mod_TrainTernaryResNetE18("cifar100")
    Mod_TrainTerRealNet("cifar10")
    Mod_TrainTerRealNet("cifar100")
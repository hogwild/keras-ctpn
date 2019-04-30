# -*- coding: utf-8 -*-
"""
   File Name：     train
   Description :   ctpn训练
   Author :       mick.yi
   date：          2019/3/14
"""
import os
import sys
import tensorflow as tf
import keras
import argparse
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from ctpn.layers import models
from ctpn.config import cur_config as config
from ctpn.utils import file_utils
from ctpn.utils.generator import generator
from ctpn.preprocess import reader


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def get_call_back():
    """
    定义call back
    :return:
    """
    checkpoint = ModelCheckpoint(filepath='/tmp/ctpn.{epoch:03d}.h5',
                                 monitor='acc',
                                 verbose=1,
                                 save_best_only=False,
                                 period=5)

    # 验证误差没有提升 verify if the loss is rising
    lr_reducer = ReduceLROnPlateau(monitor='loss',
                                   factor=0.1,
                                   cooldown=0,
                                   patience=10,
                                   min_lr=0)
    log = TensorBoard(log_dir='log')
    return [lr_reducer, checkpoint, log]


def main(args):
    set_gpu_growth()
    # 加载标注 load the annotations
    annotation_files = file_utils.get_sub_files(config.IMAGE_GT_DIR)
    image_annotations = [reader.load_annotation(file,
                                                config.IMAGE_DIR) for file in annotation_files]
    # 过滤不存在的图像，ICDAR2017中部分图像找不到 remove the missing images
    image_annotations = [ann for ann in image_annotations if os.path.exists(ann['image_path'])]
    # 加载模型 load the model
    m = models.ctpn_net(config, 'train')
    models.compile(m, config, loss_names=['ctpn_regress_loss', 'ctpn_class_loss', 'side_regress_loss'])
    # 增加度量 increasing the metrics
    output = models.get_layer(m, 'ctpn_target').output
    models.add_metrics(m, ['gt_num', 'pos_num', 'neg_num', 'gt_min_iou', 'gt_avg_iou'], output[-5:])
    if args.init_epochs > 0:
        m.load_weights(args.weight_path, by_name=True)
    else:
        m.load_weights(config.PRE_TRAINED_WEIGHT, by_name=True)
    m.summary()
    # 生成器 generator
    gen = generator(image_annotations,
                    config.IMAGES_PER_GPU,
                    config.IMAGE_SHAPE,
                    config.ANCHORS_WIDTH,
                    config.MAX_GT_INSTANCES)

    # 训练 training
    m.fit_generator(gen,
                    steps_per_epoch=len(image_annotations) // config.IMAGES_PER_GPU * 2,
                    epochs=args.epochs,
                    initial_epoch=args.init_epochs,
                    verbose=True,
                    callbacks=get_call_back(),
                    workers=2,
                    use_multiprocessing=True)

    # 保存模型 model saving
    m.save(config.WEIGHT_PATH)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--epochs", type=int, default=50, help="epochs")
    parse.add_argument("--init_epochs", type=int, default=0, help="epochs")
    parse.add_argument("--weight_path", type=str, default=None, help="weight path")
    argments = parse.parse_args(sys.argv[1:])
    main(argments)

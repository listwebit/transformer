# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging

logging.basicConfig(level=logging.INFO)

# 参数
logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
#数据预处理
# 处理训练集数据，生成一个batch
train_batches, num_train_batches, num_train_samples = get_batch(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
# 处理验证集集数据，生成一个batch
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
#实例化模型
m = Transformer(hp)
#传入x,y（label）开始训练模型
loss, train_op, global_step, train_summaries = m.train(xs, ys)
# 传入x,y（label）开始验证模型
y_hat, eval_summaries = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    #从训练log中恢复模型
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    # 如果没有模型初始化
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    # 如果有模型恢复模型
    else:
        saver.restore(sess, ckpt)
    # 指定模型保存文件路径和网络结构
    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    # 计算总batch数量  num_train_batches：训练一轮batch的数量个数
    total_steps = hp.num_epochs * num_train_batches
    # 当前batch步数
    _gs = sess.run(global_step)
    #循环batch数量开始训练，tqdm是为了显示进度条
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        print(train_op)
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)
        # 当训练一轮完成
        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses))

            # 验证模型
            logging.info("# calc bleu score and append it to translation")
            calc_bleu(hp.eval3, translation)

            # 保存模型
            logging.info("# save models")
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")

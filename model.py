# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        # 字向量(tooke向量)，将待翻译的每个字映射到目标词表中
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks：x的masks
            # [[False False False False False  True  True  True  True  True  True  True
            #   True  True  True  True  True  True  True  True  True  True  True]
            #   [False False False False False False False False False False  True  True
            #   True  True  True  True  True  True  True  True  True  True  True]
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            # 将位置信息加入
            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks

    def decode(self, ys, memory, src_masks, training=True):
        '''
        memory: encoder outputs. (N, T1, d_model)
        src_masks: (N, T1)

        Returns
        logits: (N, T2, V). float32.
        y_hat: (N, T2). int32  预测值： [[1,2133,2123,4444,...],[1,2342,43434,....]]
        y: (N, T2). int32      label值：[[1,2133,2123,4444,...],[1,2342,43434,....]]
        sents2: (N,). string.
        '''
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):

            # decoder_inputs:(N,T2) 是开头加了<s> 去掉最后一个，目的是要添加到x中预测下一个字
            # decoder_inputs：[[    2    40   496   305 25112 31943     0     0     0],[2   142  2823    77  2960   532   368  1230]]
            # y:就是label:(N,T2)
            # y:[[   40   496   305 25112 31943     3     0     0],[  142  2823    77  2960   532   368  1230 319
            # seqlens:[ 6 24 14 41]
            # sents2:(N)
            # sents2:  [b'_I _am _my _connectome .', b"_And _though _that _company _has _]
            decoder_inputs, y, seqlens, sents2 = ys

            # tgt_masks:(N,T2)
            # [[False False False False False False False False False False  True  True
            #   True  True  True  True  True  True  True  True  True  True  True  True
            #   True  True  True  True]
            #   [False False False False False False False False False  True  True  True
            #   True  True  True  True  True  True  True  True  True  True  True  True
            #   True  True  True  True]
            tgt_masks = tf.math.equal(decoder_inputs, 0)  # (N, T2)

            # embedding 将字索引id转换为向量
            # self.embeddings：字向量：(V，E)
            # dec：(N, T2, d_model)
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)  # (N, T2, d_model)
            dec *= self.hp.d_model ** 0.5  # scale
            # 加上位置张量  positional_encoding
            # dec：(N, T2, d_model)
            dec += positional_encoding(dec, self.hp.maxlen2)
            # dropout
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # Masked self-attention (Note that causality is True at this time)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              key_masks=tgt_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=True,
                                              scope="self_attention")

                    # Vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False,
                                              scope="vanilla_attention")
                    ### Feed Forward 前馈网络
                    dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # Final linear projection (embedding weights are shared)
        #将参数矩阵转置
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        #tf.einsum：爱因斯坦求和 transformer之后的字矩阵与权重矩阵相乘得到词典中每个字的概率
        #logits:(N, T2,[0,0.999,0,.....,0]
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)

        #tf.argmax：返回某维度最大的索引
        # y_hat:(N, T2） 值:[[ 5768  7128  7492  3546  7128  3947 21373  7128  7128  7128  7492  4501
        #                                      7128  7128 14651],]
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        #参考注释
        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward
        memory, sents1, src_masks = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory, src_masks)

        # train scheme
        # y:(N, T2）值:[[ 5768  7128  7492  7128  7492  4501 7128  7128 14651],[ 5768  7128  7492  7128  7492  4501 7128  7128 14651]]
        # y_:(N, T2, vocab_size);  值：(N, T2,[0,0.999,0,.....,0]
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        # 预测值和label做交叉熵，生成损失值
        # logits:预测id的概率 (N, T2,[0,0.999,0,.....,0])
        # ce: (N,T2) 例如：(4, 42)  值：array([[ 6.8254533,  6.601975 ,  6.5515084...,9.603574 , 10.001306 ],[6.8502007,  6.645137...]】,每个字粒度的损失
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        # nonpadding:(N,T2) 例如：(4, 42) 值：[[[1., 1., 1.,0,0,0],[1., 1., 1., 1., 1., 1.,.....]
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))  # 0: <pad>
        # tf.reduce_sum 按照某一维度求和 不指定axis，默认所有维度
        # ce * nonpadding:只求没有填充的词的损失，padding的去掉了 tf.reduce_sum(nonpadding)：个数相加为了求平均
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        # 根据训练步数，动态改变学习率
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        #定义优化器
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.scalar('lr', lr)


        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<s>"]
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1, src_masks = self.encode(xs, False)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, src_masks, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]: break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries


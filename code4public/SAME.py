#!/usr/bin/env python
#-*- coding:utf8 -*-
import sys
import os
cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, '..')))
import tensorflow as tf
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.ops import metrics
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from model_util.fg import FgParser
from model_util.util import *
from model_util.attention import attention as atten_func
from model_util.gate import gate_layer as gate_func
from optimizer.adagrad_decay import SearchAdagradDecay
from optimizer.adagrad import SearchAdagrad
from optimizer.gradient_decent import SearchGradientDecent
from optimizer.gradient import SearchGradient
from optimizer import optimizer_ops as myopt


class SAMENet():

    def __init__(self, context):
        self.context = context
        self.logger = self.context.get_logger()
        self.config = self.context.get_config()

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']
        self.model_param = self.model_conf['model_hyperparameter']

        self.model_name = "SAME"

        # Feature Blocks
        self.main_column_blocks = []
        self.bias_column_blocks = []
        self.cross_column_blocks = []
        self.gate_column_blocks = []
        self.expert_name_list = []

        if self.algo_config.get('main_columns') is not None:
            arr_blocks = self.algo_config.get('main_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.main_column_blocks.append(block)
        else:
            raise RuntimeError("main_columns must be specified.")

        if self.algo_config.get('bias_columns') is not None:
            arr_blocks = self.algo_config.get('bias_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.bias_column_blocks.append(block)

        if self.algo_config.get('gate_columns') is not None and self.algo_config.get('expert_columns') is not None:
            arr_blocks = self.algo_config.get('gate_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.gate_column_blocks.append(block)
            arr_blocks = self.algo_config.get('expert_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.expert_name_list.append(block)

        self.seq_column_blocks = []
        self.seq_column_len = {}
        self.seq_column_atten = {}

        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                arr = block.split(':', -1)
                if len(arr) != 4: continue
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]
                if len(arr[2]) > 0:
                    self.seq_column_atten[arr[0] + '_user'] = arr[2]
                if len(arr[3]) > 0:
                    self.seq_column_atten[arr[0] + '_item'] = arr[3]

        self.logger.info('main column blocks: {}'.format(self.main_column_blocks))
        self.logger.info('bias column blocks: {}'.format(self.bias_column_blocks))
        self.logger.info('gate column blocks: {}'.format(self.gate_column_blocks))
        self.logger.info('sequence column blocks: {}'.format(self.seq_column_blocks))
        self.logger.info('sequence column attention: {}'.format(self.seq_column_atten))

        # Define model variables collection
        self.atten_collections_dnn_hidden_layer = "{}_atten_dnn_hidden_layer".format(self.model_name)
        self.atten_collections_dnn_hidden_output = "{}_atten_dnn_hidden_output".format(self.model_name)
        self.main_collections_dnn_hidden_layer = "{}_main_dnn_hidden_layer".format(self.model_name)
        self.main_collections_dnn_hidden_output = "{}_main_dnn_hidden_output".format(self.model_name)
        self.bias_collections_dnn_hidden_layer = "{}_bias_dnn_hidden_layer".format(self.model_name)
        self.bias_collections_dnn_hidden_output = "{}_bias_dnn_hidden_output".format(self.model_name)
        self.gate_collections_dnn_hidden_layer = "{}_gate_dnn_hidden_layer".format(self.model_name)
        self.gate_collections_dnn_hidden_output = "{}_gate_dnn_hidden_output".format(self.model_name)
        self.logits_collections_dnn_hidden_layer = "{}_logits_dnn_hidden_layer".format(self.model_name)
        self.logits_collections_dnn_hidden_output = "{}_logits_dnn_hidden_output".format(self.model_name)

        self.layer_dict = {}
        self.sequence_layer_dict = {}
        self.metrics = {}
        self.fg = FgParser(self.config.get_fg_config())

        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")


    def build_graph(self, context, features, feature_columns, labels):
        self.set_global_step()
        self.inference(features, feature_columns)
        self.loss(self.logits, labels)
        self.optimizer(self.loss_op)
        self.predictions(self.logits)
        self.summary()


    def inference(self, features, feature_columns):
        self.feature_columns = feature_columns
        self.features = features
        self.embedding_layer(features, feature_columns)
        self.expert_layer()
        self.logits = self.logits_layer()
        return self.logits


    def loss(self, logits, label):
        with tf.name_scope("{}_Loss_Op".format(self.model_name)):
            self.label = label
            self.logits = logits
            self.reg_loss_f()
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label)
            self.loss_op = tf.reduce_mean(loss) + self.reg_loss
            return self.loss_op


    def predictions(self, logits):
        with tf.name_scope("{}_Predictions_Op".format(self.model_name)):
            self.prediction = tf.sigmoid(logits)
            return self.prediction


    def optimizer(self, loss_op):
        with tf.variable_scope(
                name_or_scope="Optimize",
                partitioner=partitioned_variables.min_max_variable_partitioner(
                    max_partitions=self.config.get_job_config("ps_num"),
                    min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                ),
                reuse=tf.AUTO_REUSE):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = self.update_op()

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                if 'scope' not in opt_conf or opt_conf["scope"] == "Global":
                    global_opt_name = opt_name
                    global_optimizer = optimizer
                else:
                    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=opt_conf["scope"])
                    if len(vars) != 0:
                        for var in vars:
                            if var in global_opt_vars:
                                global_opt_vars.remove(var)
                        train_op, _, _ = myopt.optimize_loss(
                            loss=loss_op,
                            global_step=self.global_step,
                            learning_rate=opt_conf.get("learning_rate", 0.01),
                            optimizer=optimizer,
                            clip_gradients=opt_conf.get('clip_gradients', 5),
                            variables=vars,
                            increment_global_step=False,
                            summaries=myopt.OPTIMIZER_SUMMARIES)
                        train_ops.append(train_op)
            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op


    def update_op(self):
        update_ops = []
        for update_op in tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            update_ops.append(update_op)
        return update_ops


    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer_dict = {
            "AdagradDecay": lambda opt_conf, global_step: SearchAdagradDecay(opt_conf).get_optimizer(global_step),
            "Adagrad": lambda opt_conf, global_step: SearchAdagrad(opt_conf).get_optimizer(global_step),
            "GradientDecentDecay": lambda opt_conf, global_step: SearchGradientDecent(opt_conf).get_optimizer(global_step),
            "GradientDecent": lambda opt_conf, global_step: SearchGradient(opt_conf).get_optimizer(global_step)
        }
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name and isinstance(optimizer_dict[name], str):
                optimizer = optimizer_dict[name]
                break
            elif opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break
        return optimizer


    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)


    def embedding_layer(self, features, feature_columns):
        with tf.variable_scope(name_or_scope="Embedding_Layer",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for block_name in (self.main_column_blocks + self.bias_column_blocks +
                               self.seq_column_len.values() + self.gate_column_blocks):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.logger.info("block_name:%s, len(feature_columns[block_name])=%d" %
                                 (block_name, len(feature_columns[block_name])))

                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[block_name],
                                                                                scope=scope)

        with tf.variable_scope(name_or_scope="seq_input_from_feature_columns",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            if len(self.seq_column_blocks) > 0:
                for block_name in self.seq_column_blocks:
                    if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                        raise ValueError("block_name:(%s) not in feature_columns for seq" % block_name)
                    seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                    sequence_layer = layers.input_from_feature_columns(features, feature_columns[block_name], scope=scope)

                    if self.model_param['atten_param']['seq_type'] == 'sum':
                        sequence_split = tf.split(sequence_layer, len(feature_columns[block_name]), axis=1)
                        sequence_stack = tf.stack(values=sequence_split)
                        sequence_layer = tf.reduce_sum(sequence_stack, axis=0)
                    sequence = tf.split(sequence_layer, seq_len, axis=0)
                    sequence_stack = tf.stack(values=sequence, axis=1)
                    sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                    if block_name in self.seq_column_len and self.seq_column_len[block_name] in self.layer_dict:
                        sequence_length = self.layer_dict[self.seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                        sequence_stack = tf.reshape(tf.where(tf.reshape(sequence_mask, [-1]),
                                                             sequence_2d, tf.zeros_like(sequence_2d)),
                                                    tf.shape(sequence_stack))
                    else:
                        sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                    # (B,N,d)
                    self.sequence_layer_dict[block_name] = sequence_stack

        with tf.variable_scope(name_or_scope="atten_input_from_feature_columns",
                               partitioner=partitioned_variables.min_max_variable_partitioner(
                                   max_partitions=self.config.get_job_config("ps_num"),
                                   min_slice_size=self.config.get_job_config("embedding_min_slice_size")
                               ),
                               reuse=tf.AUTO_REUSE) as scope:
            for atten_block_name in self.seq_column_atten.values():
                if len(atten_block_name) <= 0: continue
                if atten_block_name not in feature_columns or len(feature_columns[atten_block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for atten" % atten_block_name)
                self.layer_dict[atten_block_name] = layers.input_from_feature_columns(features,
                                                                                      feature_columns[atten_block_name],
                                                                                      scope=scope)


    def sequence_layer(self, part_name):
        seq_emb_list = []
        for block_name in self.sequence_layer_dict.keys():
            with arg_scope(model_arg_scope(
                    weight_decay=self.model_param['atten_param'].get('attention_l2_reg', 0.0)
            )):
                with tf.variable_scope(name_or_scope="{}_Sequence_Layer_{}".format(part_name, block_name),
                                       partitioner=partitioned_variables.min_max_variable_partitioner(
                                           max_partitions=self.config.get_job_config("ps_num"),
                                           min_slice_size=self.config.get_job_config("dnn_min_slice_size")),
                                       reuse=tf.AUTO_REUSE) as scope:

                    max_len = self.fg.get_seq_len_by_sequence_name(block_name)

                    sequence = self.sequence_layer_dict[block_name]
                    if block_name not in self.seq_column_len or self.seq_column_len[block_name] not in self.layer_dict:
                        sequence_mask = tf.sequence_mask(tf.ones_like(sequence[:, 0, 0], dtype=tf.int32), 1)
                        sequence_mask = tf.tile(sequence_mask, [1, max_len])
                    else:
                        sequence_length = self.layer_dict[self.seq_column_len[block_name]]
                        sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), max_len)

                    # sequence self attention
                    # vec shape: (batch, seq_len, dim)
                    dec = []
                    self.logger.info('add self attention.')
                    vec, self_atten_weight = atten_func(query_masks=sequence_mask,
                                                        key_masks=sequence_mask,
                                                        queries=sequence,
                                                        keys=sequence,
                                                        num_units=self.model_param['atten_param']['sa_num_units'],
                                                        num_output_units=self.model_param['atten_param']['sa_num_output_units'],
                                                        scope=block_name + "self_attention",
                                                        atten_mode=self.model_param['atten_param']['atten_mode'],
                                                        reuse=tf.AUTO_REUSE,
                                                        variables_collections=[self.atten_collections_dnn_hidden_layer],
                                                        outputs_collections=[self.atten_collections_dnn_hidden_output],
                                                        num_heads=self.model_param['atten_param']['num_heads'],
                                                        residual_connection=self.model_param['atten_param'].get('residual_connection', False),
                                                        attention_normalize=self.model_param['atten_param'].get('attention_normalize', False))

                    # must be given user attention blocks or item attention blocks
                    if self.seq_column_atten[block_name + '_user'] not in self.layer_dict and self.seq_column_atten[block_name + '_item'] not in self.layer_dict:
                        raise RuntimeError("No existing attention layer.")

                    # user attention
                    if self.seq_column_atten[block_name + '_user'] in self.layer_dict:
                        self.logger.info('add user attention.')
                        attention_layer = tf.concat(self.layer_dict[self.seq_column_atten[block_name + '_user']], axis=1)
                        attention = tf.expand_dims(attention_layer, 1)

                        # sequence X user intent attention
                        # user_vec shape: (batch, 1, att_out)
                        user_vec, user_atten_weight = atten_func(queries=attention,
                                                                 keys=vec,
                                                                 key_masks=sequence_mask,
                                                                 query_masks=tf.sequence_mask(tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                                                 num_units=self.model_param['atten_param']['ma_num_units'],
                                                                 num_output_units=self.model_param['atten_param']['ma_num_output_units'],
                                                                 scope=block_name + "user_multihead_attention",
                                                                 atten_mode=self.model_param['atten_param']['atten_mode'],
                                                                 reuse=tf.AUTO_REUSE,
                                                                 variables_collections=[self.atten_collections_dnn_hidden_layer],
                                                                 outputs_collections=[self.atten_collections_dnn_hidden_output],
                                                                 num_heads=self.model_param['atten_param']['num_heads'],
                                                                 residual_connection=self.model_param['atten_param'].get('residual_connection', False),
                                                                 attention_normalize=self.model_param['atten_param'].get('attention_normalize', False))

                        if self.model_param['atten_param'].get('residual_connection', False):
                            ma_num_output_units = attention.get_shape().as_list()[-1]
                        else:
                            ma_num_output_units = self.model_param['atten_param']['ma_num_output_units']
                        dec.append(tf.reshape(user_vec, [-1, ma_num_output_units]))

                    if self.seq_column_atten[block_name + '_item'] in self.layer_dict:
                        self.logger.info('add item attention.')
                        attention_layer = tf.concat(self.layer_dict[self.seq_column_atten[block_name + '_item']], axis=1)
                        attention = tf.expand_dims(attention_layer, 1)

                        # sequence X target attention
                        # item_vec shape: (batch, seq_len, att_out)
                        item_vec, item_atten_weight = atten_func(queries=attention,
                                                                 keys=vec,
                                                                 key_masks=sequence_mask,
                                                                 query_masks=tf.sequence_mask(tf.ones_like(attention[:, 0, 0], dtype=tf.int32), 1),
                                                                 num_units=self.model_param['atten_param']['ma_num_units'],
                                                                 num_output_units=self.model_param['atten_param']['ma_num_output_units'],
                                                                 scope=block_name + "item_multihead_attention",
                                                                 atten_mode=self.model_param['atten_param']['atten_mode'],
                                                                 reuse=tf.AUTO_REUSE,
                                                                 variables_collections=[self.atten_collections_dnn_hidden_layer],
                                                                 outputs_collections=[self.atten_collections_dnn_hidden_output],
                                                                 num_heads=self.model_param['atten_param']['num_heads'],
                                                                 residual_connection=self.model_param['atten_param'].get('residual_connection', False),
                                                                 attention_normalize=self.model_param['atten_param'].get('attention_normalize', False))

                        if self.model_param['atten_param'].get('residual_connection', False):
                            ma_num_output_units = attention.get_shape().as_list()[-1]
                        else:
                            ma_num_output_units = self.model_param['atten_param']['ma_num_output_units']
                        dec.append(tf.reshape(item_vec, [-1, ma_num_output_units]))
                    seq_emb_list.append(tf.concat(dec, axis=1))
        return seq_emb_list


    def expert_layer(self):
        self.expert_dict = self.expert_net()
        self.gate_dict = self.gate_net()
        self.merged_rep = tf.constant(0, dtype=tf.float32)
        for expert_name in self.expert_name_list:
            alpha = self.gate_dict[expert_name]
            logits = self.expert_dict[expert_name]
            self.merged_rep += alpha * logits


    def expert_net(self):
        expert_dict = dict()
        for i, expert_name in enumerate(self.expert_name_list):
            seq_column_blocks = self.sequence_layer(expert_name)
            main_net_layer = seq_column_blocks
            for block_name in (self.main_column_blocks):
                if not self.layer_dict.has_key(block_name):
                    raise ValueError('[Main net, layer dict] does not has block : {}'.format(block_name))
                main_net_layer.append(self.layer_dict[block_name])
                self.logger.info('[main_net] add %s' % block_name)
            main_net = tf.concat(values=main_net_layer, axis=1)
            with tf.variable_scope(name_or_scope="{}_Main_Score_Network_Part_{}".format(self.model_name, expert_name),
                                     partitioner=partitioned_variables.min_max_variable_partitioner(
                                         max_partitions=self.config.get_job_config("ps_num"),
                                         min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                     ):
                with arg_scope(model_arg_scope(weight_decay=self.model_param['dnn_l2_reg'])):
                    for layer_id, num_hidden_units in enumerate(self.model_param['dnn_hidden_units']):
                        with tf.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                            main_net = layers.fully_connected(
                                main_net,
                                num_hidden_units,
                                getActivationFunctionOp(self.model_param['activation']),
                                scope=dnn_hidden_layer_scope,
                                variables_collections=[self.main_collections_dnn_hidden_layer],
                                outputs_collections=[self.main_collections_dnn_hidden_output],
                                normalizer_fn=layers.batch_norm if self.model_param.get('batch_norm', True) else None,
                                normalizer_params={"scale": True, "is_training": self.is_training})
                    if self.model_param['need_dropout']:
                        main_net = tf.layers.dropout(
                            main_net,
                            rate=self.model_param['dropout_rate'],
                            noise_shape=None,
                            seed=None,
                            training=self.is_training,
                            name=None)
            expert_rep = main_net

            if len(self.bias_column_blocks) > 0:
                bias_net_layer = []
                for block_name in self.bias_column_blocks:
                    if not self.layer_dict.has_key(block_name):
                        raise ValueError('[Bias net, layer dict] does not has block : {}'.format(block_name))
                    bias_net_layer.append(self.layer_dict[block_name])
                    self.logger.info('[bias_net] add %s' % block_name)
                bias_net = tf.concat(values=bias_net_layer, axis=1)
                with tf.variable_scope(name_or_scope="{}_Bias_Score_Network_Part_{}".format(self.model_name, expert_name),
                                         partitioner=partitioned_variables.min_max_variable_partitioner(
                                             max_partitions=self.config.get_job_config("ps_num"),
                                             min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                         ):
                    with arg_scope(model_arg_scope(weight_decay=self.model_param['dnn_l2_reg'])):
                        for layer_id, num_hidden_units in enumerate(self.model_param['bias_dnn_hidden_units']):
                            with tf.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                                bias_net = layers.fully_connected(
                                    bias_net,
                                    num_hidden_units,
                                    getActivationFunctionOp(self.model_param['activation']),
                                    scope=dnn_hidden_layer_scope,
                                    variables_collections=[self.bias_collections_dnn_hidden_layer],
                                    outputs_collections=[self.bias_collections_dnn_hidden_output],
                                    normalizer_fn=layers.batch_norm if self.model_param.get('batch_norm', True) else None,
                                    normalizer_params={"scale": True, "is_training": self.is_training})
                        if self.model_param['need_dropout']:
                            bias_net = tf.layers.dropout(
                                bias_net,
                                rate=self.model_param['dropout_rate'],
                                noise_shape=None,
                                seed=None,
                                training=self.is_training,
                                name=None)
                expert_rep = nn_ops.bias_add(main_net, bias_net)
            expert_dict[expert_name] = expert_rep
        return expert_dict


    def gate_net(self):
        self.logger.info('add gate net.')
        gate_net_layer = []
        for block_name in self.gate_column_blocks:
            if not self.layer_dict.has_key(block_name):
                raise ValueError('[Gate net] layer dict does not has block : {}'.format(block_name))
            gate_net_layer.append(self.layer_dict[block_name])

        with tf.variable_scope(name_or_scope="{}_Gate_Score_Network".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                 max_partitions=self.config.get_job_config("ps_num"),
                                 min_slice_size=self.config.get_job_config("dnn_min_slice_size")),
                                 reuse=tf.AUTO_REUSE):
            gate_net_layer = tf.concat(values=gate_net_layer, axis=1)
            gate_dict = gate_func(gate_net_layer,
                                   self.model_param.get('gate_hidden_units'),
                                   self.expert_name_list,
                                   self.expert_dict,
                                   merge_type=self.model_param['gate_weight_type'],
                                   reuse=tf.AUTO_REUSE,
                                   outputs_collections=[self.gate_collections_dnn_hidden_output],
                                   variables_collections=[self.gate_collections_dnn_hidden_layer],
                                   normalizer_fn=layers.batch_norm if self.model_param.get('batch_norm', True) else None,
                                   normalizer_params={"scale": True, "is_training": self.is_training})
        return gate_dict


    def logits_layer(self):
        main_net = self.merged_rep
        with tf.variable_scope(name_or_scope="{}_Logits".format(self.model_name),
                                 partitioner=partitioned_variables.min_max_variable_partitioner(
                                     max_partitions=self.config.get_job_config("ps_num"),
                                     min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
                                 ):
            with arg_scope(model_arg_scope(weight_decay=self.model_param['dnn_l2_reg'])):
                for layer_id, num_hidden_units in enumerate(self.model_param['logits_hidden_units']):
                    with tf.variable_scope(name_or_scope="hiddenlayer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        main_net = layers.fully_connected(
                            main_net,
                            num_hidden_units,
                            getActivationFunctionOp(self.model_param['activation']),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.logits_collections_dnn_hidden_layer],
                            outputs_collections=[self.logits_collections_dnn_hidden_output],
                            normalizer_fn=layers.batch_norm if self.model_param.get('batch_norm', True) else None,
                            normalizer_params={"scale": True, "is_training": self.is_training})
                main_logits = layers.linear(
                    main_net,
                    1,
                    scope="logits_net",
                    variables_collections=[self.logits_collections_dnn_hidden_layer],
                    outputs_collections=[self.logits_collections_dnn_hidden_output],
                    biases_initializer=None)
                bias = contrib_variables.model_variable(
                    'bias_weight',
                    shape=[1],
                    initializer=tf.zeros_initializer(),
                    trainable=True)
                logits = nn_ops.bias_add(main_logits, bias)
        return logits


    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_tmp = []
        for reg_loss in reg_losses:
            reg_tmp.append(reg_loss)
        self.reg_loss = tf.reduce_sum(reg_tmp)


    def summary(self):
        with tf.name_scope("{}_Metrics".format(self.model_name)):
            self.logger.info('[summary] task_id={}'.format(self.context.get_task_id()))
            worker_device = "/job:worker/task:{}".format(self.context.get_task_id())
            with tf.device(worker_device):
                self.current_auc, self.total_auc = auc(labels=self.label,
                                                       predictions=self.prediction,
                                                       num_thresholds=2000,
                                                       name=self.model_name + '-auc')
                decay_rate = self.algo_config.get('auc', {}).get('decay_rate', 0.999)
                current_auc_decay, update_auc_decay = metrics.auc(
                    labels=self.label,
                    predictions=self.prediction,
                    num_thresholds=2000,
                    name=self.model_name + '-decay_auc-' + str(decay_rate),
                    decay_rate=decay_rate)
                with tf.control_dependencies([update_auc_decay]):
                    current_auc_decay = tf.identity(current_auc_decay)

        metrics_dict = {'scalar/auc': self.current_auc,
                        'scalar/total_auc': self.total_auc,
                        'scalar/decay_auc-' + str(decay_rate): current_auc_decay}
        metrics_dict['scalar/loss'] = self.loss_op
        metrics_dict['scalar/reg_loss'] = self.reg_loss
        metrics_dict['scalar/label_mean'] = tf.reduce_mean(self.label)
        metrics_dict['scalar/logits_mean'] = tf.reduce_mean(self.logits)
        metrics_dict['scalar/prediction_mean'] = tf.reduce_mean(self.prediction)
        self.metrics.update(metrics_dict)
        with tf.name_scope("{}_Metrics_Scalar".format(self.model_name)):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)
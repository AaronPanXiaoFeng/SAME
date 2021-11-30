#!/usr/bin/env python
# -*- coding:utf8 -*-
import tensorflow as tf
from tensorflow.contrib import layers


def gate_layer(inputs,
               layer_num,
               expert_name_list,
               expert_dict=None,
               merge_type=None,
               scope="GateLayer",
               reuse=None,
               activation_fn=None,
               variables_collections=None,
               outputs_collections=None,
               normalizer_fn=None,
               normalizer_params=None,
               ):
    gate_dict = dict()
    if expert_name_list is None:
        return gate_dict
    num_outputs = 1 if len(expert_name_list) <= 2 else len(expert_name_list)

    with tf.variable_scope(scope, reuse=reuse):
        for i, num_units in enumerate(layer_num):
            with tf.variable_scope("hidden_layers_{}".format(i + 1)) as layer_scope:
                inputs = layers.fully_connected(
                    inputs,
                    num_units,
                    activation_fn=activation_fn,
                    variables_collections=variables_collections,
                    outputs_collections=outputs_collections,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params,
                    scope=layer_scope
                )

        with tf.variable_scope("gate_logits") as layer_scope:
            if merge_type == "concat" and expert_dict is not None:
                gate_logits_li = []
                for expert_name in expert_name_list:
                    with tf.variable_scope("gate_logits_{}".format(expert_name)) as layer_scope:
                        expert_rep = expert_dict[expert_name]
                        hid_layer = tf.concat([inputs, expert_rep], axis=1)
                        gate_out = layers.fully_connected(
                            hid_layer,
                            1,
                            activation_fn=None,
                            variables_collections=variables_collections,
                            outputs_collections=outputs_collections,
                            scope=layer_scope
                        )
                        gate_logits_li.append(gate_out)
                gate_logits = tf.nn.softmax(tf.concat(gate_logits_li, axis=1))
                for expert_name in expert_name_list:
                    gate_dict[expert_name] = tf.split(gate_logits, gate_logits.get_shape()[-1], axis=1)[i]
            elif merge_type == "dot" and expert_dict is not None:
                gate_logits_li = []
                for expert_name in expert_name_list:
                    expert_rep = expert_dict[expert_name]
                    gate_logits_li.append(tf.reshape(tf.reduce_sum(tf.multiply(inputs, expert_rep), axis=-1), [-1, 1]))
                gate_logits = tf.nn.softmax(tf.concat(gate_logits_li, axis=1))
                for expert_name in expert_name_list:
                    gate_dict[expert_name] = tf.split(gate_logits, gate_logits.get_shape()[-1], axis=1)[i]
            else:
                gate_logits = layers.fully_connected(
                    inputs,
                    num_outputs,
                    activation_fn=tf.nn.sigmoid if num_outputs == 1 else tf.nn.softmax,
                    variables_collections=variables_collections,
                    outputs_collections=outputs_collections,
                    scope=layer_scope
                )
                if num_outputs == 1:
                    gate_dict[expert_name_list[0]] = gate_logits
                    gate_dict[expert_name_list[1]] = 1 - gate_logits
                else:
                    for expert_name in expert_name_list:
                        gate_dict[expert_name] = tf.split(gate_logits, gate_logits.get_shape()[-1], axis=1)[i]
    return gate_dict
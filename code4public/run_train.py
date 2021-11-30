from SAME import SAMENet

from requests.exceptions import ConnectionError
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError


class Run():
    def init(self, context, features, feature_columns, labels):
        self.context = context
        self.logger = self.context.get_logger()
        self.config = self.context.get_config()
        self.model = SAMENet(context)
        self.features = features
        self.feature_columns = feature_columns
        self.labels = labels

    def build(self):
        self.model.build_graph(self.context, self.features, self.feature_columns, self.labels)
        self.prediction = self.model.prediction

    def run_train(self, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.model.global_step, self.model.loss_op, self.model.metrics, self.labels]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    global_step, loss, metrics, labels = mon_session.run(run_ops, feed_dict=feed_dict)
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.model.train_ops)
                    global_step, loss, metrics, labels, _ = mon_session.run(run_ops, feed_dict=feed_dict)

                auc, totalauc = metrics['scalar/auc'], metrics['scalar/total_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, loss={}, auc={}, totalauc={} thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(loss),
                        str(auc),
                        str(totalauc),
                        str(thread_index)))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

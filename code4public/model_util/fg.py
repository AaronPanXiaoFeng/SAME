from collections import OrderedDict


FEATURE_NAME_KEY = "feature_name"
SEQUENCE_NAME_KEY = "sequence_name"
FEATURE_KEY = "features"
SEQUENCE_LENGTH_KEY = "sequence_length"
value_type_key = 'value_type'
feature_type_key = 'feature_type'
feature_name_key = 'feature_name'


class FgParser(object):
  def __init__(self, fg_config):
    self.fg_config = fg_config
    self.feature_conf_map = {}
    self.seq_feature_conf_map = {}
    self.seq_len_dict = {}
    self._parse_feature_conf(self.fg_config)

  def _parse_feature_conf(self, config):
    self._feature_conf_map = OrderedDict()
    feature_conf_list = config[FEATURE_KEY]
    for feature_conf in feature_conf_list:
      if "_comment" in feature_conf:
        continue
      feature_name = feature_conf[feature_name_key]
      if feature_conf.has_key('sequence_name'):
        self.seq_feature_conf_map[feature_name] = feature_conf
        self.seq_len_dict[feature_conf['sequence_name']] = feature_conf[SEQUENCE_LENGTH_KEY]
      else:
        self.feature_conf_map[feature_name] = feature_conf

  def get_seq_len_by_sequence_name(self, name):
    return self.seq_len_dict[name]
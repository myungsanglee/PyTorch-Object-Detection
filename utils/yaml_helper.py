import re
import yaml

'''
https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
1e-3을 str이 아닌 float으로 인식하기 위한 조치
'''
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


def load_yaml_file(file):
    with open(file, 'r') as f:
        return yaml.load(f, Loader=loader)


def get_train_configs(file):
    # default_configs = load_yaml_file('configs/default_settings.yaml')

    # if file:
    #     custom_configs = load_yaml_file(file)
    # if custom_configs:
    #     default_configs.update(custom_configs)
        
    configs = load_yaml_file(file)

    return configs


if __name__ == "__main__":
    cfg = get_train_configs("configs/cls_frostnet.yaml")
    print(cfg)
    print(cfg['optimizer_options']['lr'])

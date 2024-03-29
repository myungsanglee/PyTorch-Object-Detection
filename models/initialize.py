from torch import nn

# def weight_initialize(model):
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             # nn.init.xavier_uniform_(m.weight)
#             # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             # nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
#             # nn.init.kaiming_normal_(m.weight)
#             nn.init.kaiming_uniform_(m.weight)
#             # nn.init.normal_(m.weight, 0., 0.01)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.constant_(m.weight, 1.)
#             nn.init.constant_(m.bias, 0.)
#         elif isinstance(m, nn.Linear):
#             nn.init.normal_(m.weight, 0., 0.01)
#             nn.init.constant_(m.bias, 0.)

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
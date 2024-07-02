import importlib
import torch
import time
from thop import profile
from thop.utils import clever_format

import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16
import onnx
from onnxsim import simplify as simplify_func

tracker_name = 'et_tracker'
config_name = 'et_tracker'
tracker_module = importlib.import_module('pytracking.tracker.{}'.format(tracker_name))
param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(tracker_name, config_name))
params = param_module.parameters()

tracker_class = tracker_module.get_tracker_class()
tracker = tracker_class(params)

use_gpu = False
# torch.cuda.set_device(0)
params.device = 'cpu'
model = tracker.params.net


xs = params.image_sample_size
zs = params.image_template_size

x = torch.randn(1, 3, xs, xs)
z = torch.randn(1, 3, zs, zs)
zf = torch.randn(1, 96, 8, 8)

inputs=( zf,x)
macs1, params1 = profile(model, inputs=inputs,
                             custom_ops=None, verbose=False)
macs1, params1 = clever_format([macs1, params1], "%.3f")
print('overall macs is ', macs1)
print('overall params is ', params1)

dtype = np.float16


# zf = torch.randn(1, 96, 8, 8)
# if not params.has('device'):
#     params.device = 'cuda' 
if use_gpu:
    model = model.cuda()
    x = x.cuda()
    zf = zf.cuda()

# zf = model.template(z)

inputs=(zf, x)
inputs_onnx = {
    'zf':  np.array(zf.cpu(), dtype=dtype),
    'x': np.array(x.cpu(), dtype=dtype),
            }

output_onnx_name = 'test_net.onnx'

    
# torch.onnx.export(model, 
#     inputs,
#     output_onnx_name, 
#     input_names=[ "zf", "x"], 
#     output_names=["output"],
#     opset_version=11,
#     export_params=True,
#     # verbose=True,
#     # dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
# )

# providers = ['CUDAExecutionProvider']
# onnx_model = onnx.load("test_net.onnx")
# onnx_model, success = simplify_func(onnx_model)
# onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
# onnx.save(onnx_model, "test_net.onnx")
# onnx.save(onnx_model_fp16, "test_net_fp16.onnx")
# ort_session = ort.InferenceSession("test_net_fp16.onnx", providers=providers)

T_w = 100  # warmup
T_t = 500  # test
with torch.no_grad():
    for i in range(T_w):
        # oup = model(zf, x)
        oup = model.track(x,zf)
        # output = ort_session.run(output_names=['output'],
        #                      	input_feed=inputs_onnx,
        #                         )
    t_s = time.time()
    for i in range(T_t):
        # oup = model(zf, x)
        oup = model.track(x,zf)
        # output = ort_session.run(output_names=['output'],
        #                      	input_feed=inputs_onnx,
        #                         )
    torch.cuda.synchronize()
    t_e = time.time()
    print('speed: %.2f FPS' % (T_t / (t_e - t_s)))


print("Done")

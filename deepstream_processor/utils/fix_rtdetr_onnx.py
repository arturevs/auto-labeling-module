import onnx
from onnx import numpy_helper
import numpy as np
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else './detection_models/rtdetr_traffic_sign_v7/model.onnx'
model = onnx.load(model_path)
graph = model.graph

# Fix 1: embed im_shape and scale_factor as constants (PaddlePaddle export artifact)
shape_val = np.array([[640.0, 640.0]], dtype=np.float32)
scale_val = np.array([[1.0, 1.0]], dtype=np.float32)
graph.initializer.append(numpy_helper.from_array(shape_val, name='im_shape'))
graph.initializer.append(numpy_helper.from_array(scale_val, name='scale_factor'))
new_inputs = [inp for inp in graph.input if inp.name == 'image']
graph.ClearField('input')
graph.input.extend(new_inputs)

# Fix 2: replace helper.constant.30 shape [0, 0, 8, 32] → [-1, 8, 32]
# TensorRT interprets 0 as "copy input dim", which causes volume mismatch on 2D MatMul outputs.
# The intended op is to split embed_dim=256 into num_heads=8 × head_dim=32.
for init in graph.initializer:
    if init.name == 'helper.constant.30':
        arr = numpy_helper.to_array(init)
        if list(arr) == [0, 0, 8, 32]:
            # The downstream Transpose uses perm=[0,2,3,1] (4D), so we need [1, -1, 8, 32]
            # e.g. [400,256] → [1,400,8,32] → Transpose → [1,8,400,32]
            new_shape = np.array([1, -1, 8, 32], dtype=arr.dtype)
            new_tensor = numpy_helper.from_array(new_shape, name=init.name)
            init.CopyFrom(new_tensor)
            print(f"Fixed helper.constant.30: [0, 0, 8, 32] → [1, -1, 8, 32]")

out_path = model_path.replace('.onnx', '_fixed.onnx')
onnx.save(model, out_path)
print(f"Saved: {out_path}")

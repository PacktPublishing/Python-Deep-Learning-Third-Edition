# Variables
import numpy as np
from onnx import TensorProto, numpy_helper
from onnx.helper import make_tensor_value_info

X = make_tensor_value_info(
    name='X',
    elem_type=TensorProto.FLOAT,
    shape=[None, None])

Y = make_tensor_value_info(
    'Y', TensorProto.FLOAT, [None])

A = numpy_helper.from_array(
    np.array([0.5, -0.6], dtype=np.float32),
    name='A')

B = numpy_helper.from_array(
    np.array([0.4], dtype=np.float32),
    name='B')

from onnx.helper import make_node

# Nodes
mat_mul = make_node(
    op_type='MatMul',
    inputs=['X', 'A'],
    outputs=['XA'])
addition = make_node('Add', ['XA', 'B'], ['Y'])

# Graph
from onnx.helper import make_graph

graph = make_graph(
    nodes=[mat_mul, addition],
    name='Linear regression',
    inputs=[X],
    outputs=[Y],
    initializer=[A, B])

# Model
from onnx.helper import make_model

onnx_model = make_model(graph)
onnx_model.doc_string = 'Test model'
onnx_model.model_version = 1

# Check the model for consistency
from onnx.checker import check_model

check_model(onnx_model)

print(onnx_model)

# Inference
from onnx.reference import ReferenceEvaluator
sess = ReferenceEvaluator(onnx_model)

print(sess.run(
    output_names=None,
    feed_inputs={'X': np.random.randn(2, 2).astype(np.float32)}))

# Serialization and deserialization
with open('model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

from onnx import load
with open('model.onnx', 'rb') as f:
    onnx_model = load(f)

print(onnx_model)
import onnx
from pathlib import Path

onnx_path = Path('engines/onnx/flux_2_klein_4b_transformer_512x512.onnx')
model = onnx.load(str(onnx_path), load_external_data=False)

print('ONNX inputs (with dynamic axes):')
for inp in model.graph.input:
    shape = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value:
            shape.append(str(d.dim_value))
        else:
            shape.append(f"'{d.dim_param}'")
    print(f"  {inp.name}: [{', '.join(shape)}]")

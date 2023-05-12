import torch
import hopenet
import torchvision

#  D:/Anaconda3/envs/torch310/python.exe c:/Users/Flamingo/Desktop/毕设/code/deep-head-pose/code/remove_initializer_from_input.py --input hopenet.onnx --output  hopenet_re.onnx
# 模型加载
gpu = 0
snapshot_path = "C:\\Users\\Flamingo\\Desktop\\毕设\\code\\deep-head-pose\\output\\snapshots\\hopenet_robust_alpha1.pkl"

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
# Load snapshot
saved_state_dict = torch.load(snapshot_path)
model.load_state_dict(saved_state_dict)
model.eval().cuda(gpu)

# model.eval()

x = torch.randn(1, 3, 224, 224).to(gpu)

yaw, pitch, roll = model(x)
# print(output.shape)
with torch.no_grad():
    torch.onnx.export(
        model,                   # 要转换的模型
        x,                       # 模型的任意一组输入
        'hopenet.onnx',  # 导出的 ONNX 文件名
        opset_version=15,        # ONNX 算子集版本
        input_names=['input'],   # 输入 Tensor 的名称（自己起名字）
        output_names=['yaw', 'pitch', 'roll']  # 输出 Tensor 的名称（自己起名字）
    )


# import onnx

# # 读取 ONNX 模型
# onnx_model = onnx.load('hopenet.onnx')

# # 检查模型格式是否正确
# onnx.checker.check_model(onnx_model)

# print('无报错，onnx模型载入成功')
# print(onnx.helper.printable_graph(onnx_model.graph))

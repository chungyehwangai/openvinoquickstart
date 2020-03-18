import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
ie = IECore()
net = IENetwork(model='mobilenet-v2-1.0-224.xml', weights='mobilenet-v2-1.0-224.bin')
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
batch,channel,height,width  = net.inputs[input_blob].shape
image = cv2.imread('car.png')
cv2.imshow("input", image)
image = cv2.resize(image, (width, height))
image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
exec_net = ie.load_network(network=net, device_name='CPU')
res = exec_net.infer(inputs={input_blob: image})
idx = np.argsort(res[out_blob][0])[::-1]
for i in range(5):
    print(idx[i]+1, res[out_blob][0][idx[i]])

cv2.waitKey(3*1000)

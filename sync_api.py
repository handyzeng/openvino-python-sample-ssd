#!/usr/bin/env python3

from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import os
import time

image_path = "/opt/intel/openvino/deployment_tools/demo/car.png"

model_dir = "/opt/intel/openvino/inference_engine/samples/python_samples/object_detection_demo_ssd_async/"
model_xml = model_dir + "VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.xml"
model_bin = model_dir + "VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.bin"

plugin_dir = "/opt/intel/openvino/inference_engine/lib/intel64"
cpu_extension = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so"
plugin = IEPlugin(device="CPU", plugin_dirs=plugin_dir)
plugin.add_cpu_extension(cpu_extension)
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
n, c, h, w = net.inputs[input_blob].shape
exec_net = plugin.load(network=net, num_requests=1)


start_time = time.time()
image_number = 200
for i in range(1, 1 + image_number):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (w, h))
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    res = exec_net.infer(inputs={input_blob: image}).get("detection_out")
    print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
    duration = time.time() - start_time
    print("inferred frames: " + str(i) + ", average fps: " + str(i/duration) +"\r", end = '', flush = False)

print()

del exec_net
del net
del plugin

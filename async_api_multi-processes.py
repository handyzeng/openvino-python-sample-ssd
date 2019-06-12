#!/usr/bin/env python3
import cv2
import os
import sys
import time

import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin

from multiprocessing import Process, Queue
import multiprocessing
import threading
import queue

def async_infer_worker(exec_net, image_queue, input_blob, out_blob):
    global start_time

    current_inference, next_inference = 0, 1
    infered_images = 0
    while True:
        image = image_queue.get()
        if type(image) != np.ndarray:
                break
        exec_net.start_async(request_id=current_inference, inputs={input_blob: image})
        if exec_net.requests[next_inference].wait(-1) == 0:
            infered_images = infered_images + 1
            res = exec_net.requests[next_inference].outputs[out_blob]
            #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
            duration = time.time() - start_time
            print("inferred frames: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)
        current_inference, next_inference = next_inference, current_inference
    # one more inference result left to check
    if exec_net.requests[next_inference].wait(-1) == 0:
        infered_images = infered_images + 1               
        res = exec_net.requests[next_inference].outputs[out_blob]
        #print("infer result: label:%f confidence:%f left:%f top:%f right:%f bottom:%f" %(res[0][0][0][1], res[0][0][0][2], res[0][0][0][3], res[0][0][0][4], res[0][0][0][5], res[0][0][0][6]))
        duration = time.time() - start_time
        print("inferred frames: " + str(infered_images) + ", average fps: " + str(infered_images/duration) +"\r", end = '', flush = False)
        
# for test purpose only
image_number = 200

def image_process_worker(image_queue, n, c, h, w):
    for i in range(1, 1 + image_number):
        image = cv2.imread("/opt/intel/openvino/deployment_tools/demo/car.png")
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        image_queue.put(image)
    image_queue.put(None)

start_time = -1
         
def main():
    global start_time

    image_queue = multiprocessing.Queue(maxsize= 4)

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
    exec_net = plugin.load(network=net, num_requests=2)

    start_time = time.time()

    preprocess_process = None
    preprocess_process = multiprocessing.Process(target=image_process_worker, args=(image_queue, n, c, h, w))
    preprocess_process.start()
    
    async_infer_worker(exec_net, image_queue, input_blob, out_blob)
    
    preprocess_process.join()
    print()
    
    del exec_net
    del net
    del plugin


if __name__ == '__main__':
    sys.exit(main() or 0)

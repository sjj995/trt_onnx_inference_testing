# -*- coding: utf-8 -*-
from xmlrpc.client import boolean
import onnx
import onnxruntime
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
from pathlib import Path
import os
import math
import time
from datetime import timedelta
import datetime
import tqdm
import warnings


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx_model_path",
        type=str,
        help="Onnx Model Path",
        default="/datadrive/tensorrt/onnx_models"
    )

    parser.add_argument(
        "--trt_model_save_path",
        type=str,
        help="TensorRT model save path",
        default=os.getcwd()
    )

    parser.add_argument(
        "--save_trt_model",
        type=boolean,
        default=False,
        help="decide save TensorRT model(T/F)"
    )

    return parser.parse_args()


def get_onnx_files(root_path):
    onnx_list = []
    for (root, dirs, files) in os.walk(root_path):
        for file_name in files:
           if file_name.endswith(".onnx"):
                file_path = os.path.join(root,file_name)
                onnx_list.append(file_path)
    return onnx_list


def MakeDataset(input_shape):
    batch_size = 1 if input_shape[0] == 'batch_size' else 1 if input_shape[0] == 'None' else input_shape[0]
    dim2 = input_shape[1]
    dim3 = input_shape[2]
    dim4 = input_shape[3]
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    input_data = torch.randn(batch_size,dim2,dim3,dim4).to(device)

    return input_data

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class Comparision(object):
    def __init__(self,onnx_output,trt_output):
        self.onnx_output = onnx_output
        self.trt_output = trt_output
        self.atol = 1e-06
        self.rtol = 1e-05
        self.rel_diff_propotion = None
        self.abs_diff_propotion = None
        self.max_value = -math.inf
        self.max_pos = None
        self.do_comparsion()

    def do_comparsion(self):
        self.max_value = abs(self.onnx_output-self.trt_output).max()
        self.max_pos = np.where(abs(self.onnx_output-self.trt_output) == self.max_value)

        rel_diff = np.isclose(self.onnx_output,self.trt_output,rtol=self.rtol)
        self.rel_diff_propotion = ((rel_diff == True).sum() / np.size(rel_diff))

        abs_diff = np.isclose(self.onnx_output,self.trt_output,atol=self.atol)
        self.abs_diff_propotion = ((abs_diff == True).sum() / np.size(abs_diff))
  
    def __str__(self):
#         return f"""Tolerance : [abs={self.atol}, rel={self.rtol}]
# Absolute Difference : {round(self.abs_diff_propotion*100.0,4)}% || Relative Difference : {round(self.rel_diff_propotion*100.0,4)}%
# Max Difference : {self.max_value} || Max Difference Position {self.max_pos[0][0],self.max_pos[1][0]}"""
        return f"""Tolerance : [abs={self.atol}, rel={self.rtol}]
Absolute Difference : {round(self.abs_diff_propotion*100.0,4)}% || Relative Difference : {round(self.rel_diff_propotion*100.0,4)}%
Max Difference : {self.max_value} || Max Difference Position {self.max_pos[0][0],self.max_pos[1][0]}
================================================================================="""

    def __repr__(self):
        return self.__str__()


class EvaluateOnnx(object):
    def __init__(self,onnx_model):
        self.onnx_model = onnx_model

    def remove_initializer_from_input(self):
        args = get_args()

        model = onnx.load(self.onnx_model)
        if model.ir_version < 4:
            print("Model with ir_version below 4 requires to include initilizer in graph input")
            return

        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

        #여기서 save 안하고 바로 넘기는 방법도 생각해봐야됨.
        onnx.save(model,self.onnx_model)
        #return model

    def do_onnx_inference(self):
        
        #onnx_model = self.remove_initializer_from_input()
        self.remove_initializer_from_input()
        session = onnxruntime.InferenceSession(self.onnx_model,providers=['CPUExecutionProvider','CUDAExecutionProvider','TensorrtExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_shape = session.get_outputs()[0].shape

        x = MakeDataset(input_shape)

        ort_inputs = {session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = session.run(None, ort_inputs)
     
        return input_name,input_shape,output_shape,x,ort_outs[0]


class OnnxToTensorRT(object):

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(self,input_name,input_shape,output_shape,onnx_model,trt_model_save_path,tensorrt_file_name,trt_save_flag):
        self.input_name = input_name
        self.input_shape = tuple(1 if dim == 'batch_size' or dim == 'None' else dim for dim in input_shape)
        self.output_shape = tuple(1 if dim == 'batch_size' or dim == 'None' else dim for dim in output_shape)
        self.onnx_model = onnx_model
        self.trt_model_save_path = trt_model_save_path
        self.tensorrt_file_name = tensorrt_file_name
        self.trt_save_flag = trt_save_flag
    
    # The Onnx path is used for Onnx models.
    def build_engine_onnx(self,):
        builder = trt.Builder(OnnxToTensorRT.TRT_LOGGER)
        profile = builder.create_optimization_profile()
        profile.set_shape(self.input_name, self.input_shape,self.input_shape,self.input_shape)  
        network = builder.create_network(OnnxToTensorRT.EXPLICIT_BATCH)
        config = builder.create_builder_config()
        config.add_optimization_profile(profile)
        #config.max_workspace_size = 8 << 30
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30) # Due to tensorrt 8.4 DeprecationWarning
        parser = trt.OnnxParser(network, OnnxToTensorRT.TRT_LOGGER)

        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(self.onnx_model, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        return builder.build_engine(network, config=config)
        #return builder.build_serialized_network(network,config=config) # Due to tensorrt 8.4 DeprecationWarning

    def export_engine(self):
        trt_engine = self.build_engine_onnx()
        if self.trt_save_flag == False:
            return trt_engine
        else:
            buf = trt_engine.serialize()
            with open(self.trt_model_save_path+'/'+self.tensorrt_file_name,'wb') as f:
                f.write(buf)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class EvaluateTRT(object):

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # due to cudnn version <- have to upgrade 8.4.1? now 8.4.0
    # NetworkDefinitionCreationFlag.EXPLICIT_BATCH <- 0 
    # create builder <- optimization profile, Takes a network in TensorRT and generates an engine that is optimized for the target platform. 
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)\
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    

    def __init__(self,input_shape,output_shape,onnx_model,trt_model_save_path,tensorrt_file_name,input_data,trt_save_flag,engine):
        self.input_shape = tuple(1 if dim == 'batch_size' or dim == 'None' else dim for dim in input_shape)
        self.output_shape = tuple(1 if dim == 'batch_size' or dim == 'None' else dim for dim in output_shape)
        self.onnx_model = onnx_model
        self.tensorrt_model = trt_model_save_path+'/'+tensorrt_file_name  
        self.input_data = input_data
        self.trt_save_flag = trt_save_flag
        self.trt_engine = engine
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream() 
        

    def deserialize_engine(self):
        if self.trt_save_flag == True:
            with open(self.tensorrt_model, 'rb') as f:
                engine_data = f.read()
                engine = EvaluateTRT.trt_runtime.deserialize_cuda_engine(engine_data)
        else:
            engine = self.trt_engine
        
        self.context = engine.create_execution_context()

        for binding in engine:
    
            size = trt.volume(engine.get_binding_shape(binding)) * EvaluateTRT.EXPLICIT_BATCH
            dtype = trt.nptype(engine.get_binding_dtype(binding)) 
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        input_data = to_numpy(self.input_data)
        for input in self.inputs:
            np.copyto(input.host,np.ravel(input_data))

    # This function is generalized for multiple inputs/outputs for full dimension networks.
    # inputs and outputs are expected to be lists of HostDeviceMem objects.
    def do_inference_v2(self):
        self.deserialize_engine()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        # output shape 를 어떻게 함?
        #return [out.host for out in self.outputs][0][np.newaxis,:]
        #trt_outputs =  [out.host for out in self.outputs]
        trt_outputs = [output.reshape(shape) for output, shape in zip([out.host for out in self.outputs],[self.output_shape])]
        return trt_outputs[0]
  

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    args = get_args()
    
    onnx_root_path = args.onnx_model_path
    onnx_models = get_onnx_files(onnx_root_path)
    trt_model_save_path = args.trt_model_save_path

    for om in onnx_models:
        tensorrt_file_name = Path(om).stem+'.trt'
        trt_save_flag = args.save_trt_model

        start = time.process_time()
        EvalOnnx = EvaluateOnnx(om)
        input_name,input_shape,output_shape,onnx_input,onnx_output = EvalOnnx.do_onnx_inference()
        end = time.process_time()
        print('\033[96m'+"[COMPLETE] ONNX Model Inference :", end - start,"(s)"+'\033[0m')

        start = time.process_time()
        OnnxToTrt = OnnxToTensorRT(input_name,input_shape,output_shape,om,trt_model_save_path,tensorrt_file_name,trt_save_flag)
        engine = OnnxToTrt.export_engine()
        end = time.process_time()
        print('\033[96m'+"[COMPLETE] ONNX Model to TensorRT Model :", end - start,"(s)"+'\033[0m')

        start = time.process_time()
        EvalTrt = EvaluateTRT(input_shape,output_shape,om,trt_model_save_path,tensorrt_file_name,onnx_input,trt_save_flag,engine)
        trt_output = EvalTrt.do_inference_v2()
        end = time.process_time()
        print('\033[96m'+"[COMPLETE] TensorRT Model Inference :", end - start,"(s)"+'\033[0m')

        res = Comparision(onnx_output,trt_output)
        now = datetime.datetime.now()
        print('\033[32m'+'[Final Report] : ['+Path(om).stem+' - '+(now.strftime("%c"))+']'+'\033[0m')
        print(res)
    
    return



if __name__ == "__main__":

    main()

#/bin/bash
code_dir=/data2/user/sjj995/TensorRT/build/out
src_dir=/data2/user/sjj995/TensorRT/onnx_models
log_dir=/data2/user/sjj995/TensorRT/trtexec_logs
reports_dir=/data2/user/sjj995/TensorRT/onnx_reports
trt_dir=/data2/user/sjj995/TensorRT/trt_models


find $src_dir/ -name "*.onnx" | sed "s:$src_dir/::g" |
while read -r f;
do
    dirpath=$(dirname "$f")
    onnxname=$(basename "$f")
    trtname="${onnxname%.*}"
    echo working with "$trtname"

    $code_dir/trtexec --onnx=$src_dir/$onnxname \
    --saveEngine=$trt_dir/"$trtname".trt \
    --verbose --memPoolSize=workspace:16384 --profilingVerbosity=detailed \
    --exportLayerInfo=$reports_dir/"$trtname"_layers.json \
    --exportProfile=$reports_dir/"$trtname"_profile.json > $log_dir/"$trtname"_log.txt
done

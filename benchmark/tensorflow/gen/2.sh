rm -rf /tmp/imagenet_train/
rm -f /tmp/worker*
export PYTHONPATH='/tmp//inception'


CUDA_VISIBLE_DEVICES='' python /tmp/tf_cnn_benchmarks.py --model=inception3 --batch_size=16 --ps_hosts=10.0.0.177:2222,10.0.1.107:2222 --worker_hosts=10.0.0.177:2230,10.0.1.107:2230 --job_name=ps --task_id=1 --num_batches=30 --data_format=NCHW --display_every=2 --weak_scaling=true --parameter_server=cpu --device=gpu  > /tmp/ps1 2>&1 &

python /tmp/tf_cnn_benchmarks.py --model=inception3 --batch_size=16 --ps_hosts=10.0.0.177:2222,10.0.1.107:2222 --worker_hosts=10.0.0.177:2230,10.0.1.107:2230 --job_name=worker --task_id=1 --num_batches=30 --data_format=NCHW --display_every=2 --weak_scaling=true --parameter_server=cpu --device=gpu --num_gpus=8  > /tmp/worker1 2>&1 &


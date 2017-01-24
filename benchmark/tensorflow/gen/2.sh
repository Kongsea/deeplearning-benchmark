rm -rf /tmp/imagenet_train/
rm -f /tmp/worker*
export PYTHONPATH='/tmp//inception'


CUDA_VISIBLE_DEVICES='' python imagenet_distributed_train.py --ps_hosts=10.0.0.38:2222,10.0.1.59:2222 --worker_hosts=10.0.0.38:2230,10.0.1.59:2230 --job_name=ps --task_id=1 > /tmp/ps1 2>&1 &

python imagenet_distributed_train.py --num_gpus=8 --batch_size=8 --data_dir=notused --ps_hosts=10.0.0.38:2222,10.0.1.59:2222 --worker_hosts=10.0.0.38:2230,10.0.1.59:2230 --job_name=worker --task_id=1 > /tmp/worker1 2>&1 &


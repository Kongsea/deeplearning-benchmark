rm -rf /tmp/imagenet_train/
rm -f /tmp/worker*
export PYTHONPATH='/tmp//inception'


CUDA_VISIBLE_DEVICES='' python imagenet_distributed_train.py --ps_hosts=10.0.0.218:2222 --worker_hosts=10.0.0.218:2230 --job_name=ps --task_id=0 > /tmp/ps0 2>&1 &

python imagenet_distributed_train.py --num_gpus=4 --batch_size=32 --data_dir=notused --ps_hosts=10.0.0.218:2222 --worker_hosts=10.0.0.218:2230 --job_name=worker --task_id=0 > /tmp/worker0 2>&1 &


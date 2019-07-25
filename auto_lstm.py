# Author: Duan

import time
import os
import tensorflow as tf
flag = True
while flag:
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    i = 0
    for usage in memory_gpu:
        print(usage)  # Waiting for idle GPU ;)
        if usage > 5000:
            flag = False
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            set_session(tf.Session(config=config))
            import lstm_try
            lstm_try.run(False)
        i += 1
    os.system('rm tmp')

    time.sleep(30)

import subprocess
import time
import os
import tensorflow as tf

while True:
    pipe = subprocess.Popen("~/gpu-util", shell=True, stdout=subprocess.PIPE)
    content = pipe.communicate()
    ret = content[0].decode('utf-8').split('\n')
    i = 0
    for line in ret:
        usage = int(line[41:-1])
        if usage <= 10:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            set_session(tf.Session(config=config))
            import lstm_try
            lstm_try.run()
            break
        i += 1

    time.sleep(30)

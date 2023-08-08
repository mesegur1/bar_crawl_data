import os

# Encoder options
USE_LEVEL_ENCODER = 0
USE_RBF_ENCODER = 1
USE_SINUSOID_NGRAM_ENCODER = 2
USE_GENERIC_ENCODER = 3
ENCODER_MAX = 4

# Learning mode options
USE_ADD = 0
USE_ADAPTHD = 1
USE_ONLINEHD = 2
USE_ADJUSTHD = 3
USE_NEURALHD = 4
MODE_MAX = 5

lrate_inc = 0.1
LRATE_MAX_ITER = 9
TRAIN_EPOCHS = 5

script_format_str = "python ./hdc_combined_experiment.py"

def test_suite_run():
    for i in range(ENCODER_MAX):
        for j in range(MODE_MAX):
            for k in range(LRATE_MAX_ITER):
                lrate = (k + 1) * lrate_inc
                os.system(script_format_str + " -e %d -m %d -t %d -l %.5f" % (i, j, TRAIN_EPOCHS, lrate))

if __name__ == "__main__":
    test_suite_run()
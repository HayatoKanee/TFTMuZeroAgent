# import AI_interface
import TestInterface.test_ai_interface as TestInterface
import config
from UnitTests.BaseUnitTest import runTest
import argparse
import AI_interface
import os
import time
import tensorflow as tf 

def main():
    if config.RUN_UNIT_TEST:
        runTest()

    # TODO(lobotuerk) A lot of hardcoded parameters should be used like this instead.
    parser = argparse.ArgumentParser(description='Train an AI to play TFT',
                                     epilog='For more information, '
                                            'go to https://github.com/silverlight6/TFTMuZeroAgent')
    parser.add_argument('--starting_episode', '-se', dest='starting_episode', type=int, default=0,
                        help='Episode number to start the training. Used for loading checkpoints, '
                             'disables loading if = 0')
    args = parser.parse_args()

    # interface = AI_interface.AIInterface()
    # interface.train_torch_model(starting_train_step=args.starting_episode)
    # interface.collect_dummy_data()
    # interface.testEnv()
    # interface.PPO_algorithm()
    now = time.localtime()
    subdir = time.strftime("%d-%b-%Y_%H.%M.%S", now)

    summary_dir1 = os.path.join(subdir, "t1")
    summary_writer1 = tf.summary.create_file_writer(summary_dir1)
    step = 0 
    while True:
        test_interface = TestInterface.AIInterface()
        data = test_interface.evaluate(0)
        with summary_writer1.as_default():
            for i in range(8):
                tf.summary.scalar(name="traits used", data=data[i]["traits used"] ,step=step)
                tf.summary.scalar(name="xp", data=[i]["xp bought"], step=step)
                tf.summary.scalar(name="champs", data=data[i]["champs bought"] ,step=step)
                tf.summary.scalar(name="2*s ", data=[i]["2* champs"], step=step)
                tf.summary.scalar(name="3*s ", data=[i]["3* champs"], step=step)
        step += 1 

        # except:
        #     print("Err")

    # test_interface = TestInterface.AIInterface()
    # test_interface.train_model(starting_train_step=args.starting_episode)


if __name__ == "__main__":
    main()

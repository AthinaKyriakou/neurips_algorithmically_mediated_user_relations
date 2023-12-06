# To execute /opt/miniconda3/envs/user_intercon_env/bin/python main.py --flagfile=flagfile.cfg

from absl import app
from absl import flags
import sys
sys.path.append('./RecSys2019_DeepLearning_Evaluation/')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import generate_trainset_testset
import algorithm_hyperparameter_tuning
import check_independence_assumption
import remove_n_train

FLAGS = flags.FLAGS
flags.DEFINE_string('col_user', 'UserId', help="Column name used for dataframe operations")
flags.DEFINE_string('col_item', 'ItemId', help="Column name used for dataframe operations")
flags.DEFINE_string('col_rating', 'Rating', help="Column name used for dataframe operations")
flags.DEFINE_string('col_timestamp', 'Timestamp', help="Column name used for dataframe operations")
flags.DEFINE_string('col_interaction', 'Interaction', help="Column name used for dataframe operations")
flags.DEFINE_string('col_prediction', 'Prediction', help="Column name used for dataframe operations")

#TODO: check if used
#flags.DEFINE_string('col_num_ratings', 'num_ratings', help="Column name used for dataframe operations") 
#flags.DEFINE_string('col_rec', 'recommendations', help="Column name used for dataframe operations")

# flags to be specified in the flagfile
flags.DEFINE_enum('dataset', 'MovieLens_100k', ['MovieLens_100k', 'MovieLens_1M'], help="Specify the dataset to be used in the flagfile. Dflt: 'MovieLens_100k")
flags.DEFINE_float('ratio_split_train', 0.8, help="Specify the desired train-test splitting ratio in the flagfile. Dflt: 0.8", lower_bound=0, upper_bound=1)
flags.DEFINE_float('ratio_split_validation', 0.9, help="Specify the desired train-validation splitting ratio in the flagfile. Dflt: 0.9", lower_bound=0, upper_bound=1)
flags.DEFINE_integer('random_seed', 28, help="Specify the desired train-validation splitting ratio in the flagfile")

flags.DEFINE_enum('algorithm', 'UserKNNCFRecommender', ['UserKNNCFRecommender', 'MatrixFactorization_FunkSVD_Cython'], help="Specify the CF algorithm to be used. Dflt: 'UserKNNCFRecommender")
flags.DEFINE_enum('operation', 'get_dataset_statistics', ['get_dataset_statistics','generate_trainset_testset', 'algorithm_hyperparameter_tuning', 'compute_individual_influences',  'compute_group_influences', 'check_independence_assumption'], help="Specify the operation you want to purse")

flags.DEFINE_enum('check_independence_type', 'scatter_plot', ['scatter_plot', 'hist', 'significance_testing', 'correlations'], help="Specify how you want to check the independence assumption. Dflt: 'scatter_plot") 
flags.DEFINE_list('algo_list', [], help="Specify a list of algorithms")
flags.DEFINE_list('group_sizes', [], help="Specify size of groups")
flags.DEFINE_list('colors', [], help="Specify colors list to be used in the scatter plot")
flags.DEFINE_list('markers', [], help="Specify markers list to be used in the scatter plot")

def main(argv):
    flags.FLAGS(sys.argv)
    print("in main")
    if FLAGS.operation == 'generate_trainset_testset':
        print("Generating train, test, and validation sets according to the specified ratios.")
        generate_trainset_testset.generate_trainset_testset(FLAGS.dataset)
    elif FLAGS.operation == 'algorithm_hyperparameter_tuning':
        print("Tuning algorithm %s for dataset %s." % (FLAGS.algorithm, FLAGS.dataset))
        algorithm_hyperparameter_tuning.algorithm_hyperparameter_tuning(FLAGS.dataset, FLAGS.algorithm)

    elif FLAGS.operation == 'compute_individual_influences':
        print("Computing each user's influence for algorithm %s and dataset %s." % (FLAGS.algorithm, FLAGS.dataset))    
        remove_n_train.remove_n_train(FLAGS.dataset, FLAGS.algorithm)

    elif FLAGS.operation == 'check_independence_assumption':
        print("Checking independence assumption by %s" % FLAGS.check_independence_type)
        if len(FLAGS.group_sizes) != len(FLAGS.colors) or len(FLAGS.group_sizes) != len(FLAGS.markers):
            print("The number of specified group sizes, colors, and markers should be the same!")
        else:
            check_independence_assumption.check_independence(FLAGS.check_independence_type, FLAGS.algo_list, FLAGS.dataset, FLAGS.group_sizes, FLAGS.colors, FLAGS.markers)

if __name__ == '__main__':
  app.run(main)
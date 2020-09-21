from pytorch_lightning import Trainer
import pickle
from pytorch_lightning.loggers import TensorBoardLogger
from test_tube import Experiment, SlurmCluster, HyperOptArgumentParser
from time import sleep
import pandas as pd
import numpy as np
import os
from pl_model import CaptionGenerator, FlickrDataModule
from utilities import get_dataset
import torchvision.transforms as transforms

def main(hparams, cluster):
    '''
    Once we receive this round's parameters we can load our model and our datamodule. We kept the trainer's parameters fixed for convenience and avoided using functionality that would make it hard to compare between models.
    '''

    dm = FlickrDataModule(batch_size = hparams.batch_size, num_workers = hparams.num_workers)
    dm.setup()
    # each trial has a separate version number which can be accessed from the cluster
    train, pad_idx = get_dataset(
                            "../../data/flickr8k/images",
                            "../../data/flickr8k/training_captions.txt",
                            dm.transform)
    vocab_size = len(train.vocab)
    
    # loading our model with this run's parameters
    model = CaptionGenerator(embed_size = hparams.embed_size,
                            hidden_size = hparams.hidden_size,
                            vocab_size = vocab_size,
                            num_layers =  hparams.num_layers,
                            batch_size = hparams.batch_size,
                            pad_idx = pad_idx)





    logger = TensorBoardLogger(save_dir = '../../data/caption_generator/', version = cluster.hpc_exp_number, name = 'lightning_logs')

    trainer = Trainer(logger = logger,
                    gpus = 2,
                    num_nodes = 13,
                    max_epochs = 1000,
                    auto_select_gpus = True,
                    profiler = True,
                    distributed_backend='ddp',
                    early_stop_callback=False)

    trainer.fit(model, dm)

def optimize_on_cluster(hyperparams):
    '''
    This function is in charge of creating the slurm bash scripts that will send our task to the cluster.
    For a reference single script check pl_submit.sh, located in this same folder.
    '''


    # enable cluster training
    # log all scripts to the test tube folder
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.slurm_log_path,
    )

    cluster.add_slurm_cmd(cmd = 'partition', value = 'gpu2', comment = 'queue')

    cluster.add_slurm_cmd(cmd = 'ntasks-per-node', value='2', comment = 'Tasks per node')
    cluster.job_time = '0-17:00:00'

    # email for cluster coms
    cluster.add_slurm_cmd('mail-type', value = 'all', comment = 'Mail type')
    cluster.add_slurm_cmd('mail-user', value = 'Rodrigo.Lopez@mpikg.mpg.de', comment = 'Mail account')

    # configure cluster
    cluster.per_experiment_nb_gpus = 2
    cluster.per_experiment_nb_nodes = 13

    cluster.memory_mb_per_node = 0

    # any modules for code to run in env
    cluster.add_command('module purge')
    cluster.add_command('module load python/3.8.2')
    cluster.add_command('module load nvidia/cuda/9.1')

    cluster.add_command('set')


    # run hopt
    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=24,
        job_name='grid_test'
    )

if __name__ ==  '__main__':

    # subclass of argparse
    parser = HyperOptArgumentParser(strategy='grid_search')

    #  params we don't want to search over. will use the default value
    parser.add_argument('--slurm_log_path', default=os.getcwd(), type=str, help='Save experiments here')
    parser.add_argument('--vocab_size', default=3432, type=int)
    parser.add_argument('--embed_size', default=150, type=int)
    parser.add_argument('--hidden_size', default=250, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    # params we want to search over

    #parser.opt_list('--embed_size', default=100, type=int, tunable=True, options=[150, 200, 250, 300])
    #arser.opt_list('--hidden_size', default=100, type=int, tunable=True, options=[100, 200, 300])
    #parser.opt_list('--num_layers', default=1, type=int, tunable=True, options=[1, 2])
    #parser.opt_list('--batch_size', default=32, type=int, tunable=True, options=[32, 64, 128])
    #parser.opt_list('--num_workers', default=15, type=int, tunable=True, options=[15])

    # compile (because it's argparse underneath)
    hparams = parser.parse_args()

    optimize_on_cluster(hparams)

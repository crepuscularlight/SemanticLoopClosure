import yaml
import os

class model_config:
    def __init__(self):
        self.is_setup=False

    def load(self,config_file):
        configs=yaml.full_load(open(os.path.abspath(config_file)))
        #common
        self.model=configs["common"]["model"]
        self.dataset=configs["common"]["dataset"]
        self.trainer=configs["common"]["trainer"]
        self.exp_name=configs["common"]["exp_name"]
        self.ckpt_path=configs["common"]["ckpt_path"]
        self.graphs_dir=configs["common"]["graphs_dir"]
        self.pairs_dir=configs["common"]["pairs_dir"]
        self.eval_pairs_dir=configs["common"]["eval_pairs_dir"]

        #arch
        self.filters_dim=configs["arch"]["filters_dim"]
        self.tensor_neurons=configs["arch"]["tensor_neurons"]
        self.bottle_neck_neurons=configs["arch"]["bottle_neck_neurons"]
        self.K=configs["arch"]["K"]

        #train
        self.epoch=configs["train"]["epoch"]
        self.batch_size=configs["train"]["batch_size"]
        self.train_sequences= configs["train"]["train_sequences"]
        self.eval_sequences=configs["train"]["eval_sequences"]
        self.dropout=configs["train"]["dropout"]
        self.learning_rate=configs["train"]["learning_rate"]
        self.weight_decay=configs["train"]["weight_decay"]
        self.node_num=configs["train"]["node_num"]
        self.number_of_labels=configs["train"]["number_of_labels"]
        self.geo_output_channels=configs["train"]["geo_output_channels"]
        self.p_thresh=configs["train"]["p_thresh"]

        #test
        self.test_pairs_dir=configs["test"]["test_pairs_dir"]
        self.test_sequences=configs["test"]["test_sequences"]
        self.pair_file=configs["test"]["pair_file"]


if __name__=="__main__":
    a=model_config()
    a.load("./sgpr_baseline.yml")
    print(a.graphs_dir)

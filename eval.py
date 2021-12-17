import keras
import h5py
import numpy as np
import os.path
import tensorflow as tf
from strip import STRIP

"""
DataLoader class is inspired by https://github.com/csaw-hackml/CSAW-HackML-2020/blob/master/eval.py
"""
class DataLoader:
  def __init__(self, file_path):
    self.file_path = file_path
    self.load()
    self.preprocess()

  def load(self):
    data = h5py.File(self.file_path, "r")
    x_data = np.asarray(data["data"])
    self.x = x_data.transpose((0,2,3,1))
    self.y = np.asarray(data["label"])
  
  def preprocess(self):
    self.x = np.asarray(self.x/255, np.float64)

def num_to_net(num):
    numbers = {
      "1": {"model": "models/sunglasses_bd_net.h5", "entropy": "entropy/entropy_clean_sunglasses.h5", "fine_pruned_model": "finePruning/sunglasses_repaired_net.h5"},
      "2": {"model": "models/anonymous_1_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous1.h5", "fine_pruned_model": "finePruning/anonymous_1_repaired_net.h5"},
      "3": {"model": "models/anonymous_2_bd_net.h5", "entropy": "entropy/entropy_clean_anonymous2.h5", "fine_pruned_model": "finePruning/anonymous_2_repaired_net.h5"},
      "4": {"model": "models/multi_trigger_multi_target_bd_net.h5", "entropy": "entropy/entropy_clean_multi.h5", "fine_pruned_model": "finePruning/multi_trigger_multi_target_repaired_net.h5"},
    }
    return numbers.get(num, {"model": "models/sunglasses_bd_net.h5", "entropy": "entropy/entropy_clean_sunglasses.h5", "fine_pruned_model": "finePruning/sunglasses_repaired_net.h5"})

def num_to_mode(num):
    numbers = {
      "1": "seperate",
      "2": "mixed",
    }
    return numbers.get(num, "seperate")

def main():
    np.seterr(divide = 'ignore',invalid = 'ignore')
    # Select a test BadNet
    print('\033[0;32m' + "Select a test net: \n1. sunglasses_bd_net \n2. anonymous_1_bd_net \n3. anonymous_2_bd_net \n4. multi-trigger_multi_target_bd_net")
    net = num_to_net(input())
    model = net["model"]
    print("{0} selected!\n".format(model))
    bd_model = keras.models.load_model(model)

    # Select the test mode
    print("Select the test mode: \n1. Seperate data (clean / poisoned) \n2. Mixed data")
    test_mode = num_to_mode(input())
    if test_mode == "seperate":
        print("Seperate mode selected!\n")
        # Set the test poisoned data
        print("Please put the poisoned data under data/ and name the file poisoned_data.h5 (i.e. data/poisoned_data.h5) \nThen click enter.")
        input()
        poisoned_data_test_filename = "data/poisoned_data.h5"
        while not os.path.isfile(poisoned_data_test_filename):
            print('\033[91m' + "Error: data/poisoned_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        # Set the test clean data
        print('\033[0;32m' + "Please put the clean data under data/ and name the file clean_data.h5 (i.e. data/clean_data.h5) \nThen click enter.")
        input()
        clean_data_test_filename = "data/clean_data.h5"
        while not os.path.isfile(poisoned_data_test_filename):
            print('\033[91m' + "Error: data/clean_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        test_clean = DataLoader(clean_data_test_filename)
        test_poisoned = DataLoader(poisoned_data_test_filename)

        # Step one: STRIP
        print('\033[0;32m' + "STRIP: running......(maybe several minutes, even hours)")
        entropy_filename = net["entropy"]
        entropy_clean_data = h5py.File(entropy_filename, "r")
        entropy_clean = np.asarray(entropy_clean_data["data"])
        STRIP_filter = STRIP(50, 1.5, 1, 0)
        pred_poisoned = STRIP_filter.predict(entropy_clean, test_poisoned.x, bd_model)
        succ_att_rate = np.mean(np.equal(pred_poisoned, test_poisoned.y)) * 100
        print('\033[95m' + "Success attack rate after STRIP: {0}%".format(succ_att_rate))
        pred_clean = STRIP_filter.predict(entropy_clean, test_clean.x, bd_model)
        I_poisoned = np.argwhere(pred_poisoned == 1).ravel()
        for i in I_poisoned:
            pred_poisoned[i] = -1

        # Step two: Fine-pruning
        print('\033[0;32m' + "Fine-pruning: running......")
        fine_pruned_model = keras.models.load_model(net["fine_pruned_model"])
        I_poisoned_remaining = np.argwhere(pred_poisoned == 0).ravel()
        if I_poisoned_remaining.size != 0:
            test_poisoned_remaining = test_poisoned.x.take(I_poisoned_remaining, axis = 0)
            pred_poisoned_remaining = fine_pruned_model.predict(test_poisoned_remaining)
            succ_att_rate = (np.sum(pred_poisoned_remaining == test_poisoned.y[I_poisoned_remaining]) / test_poisoned.y.shape[0]) * 100
            for i, newpred in enumerate(pred_poisoned_remaining):
                pred_poisoned[I_poisoned_remaining[i]] = newpred
        print('\033[95m' + "Success attack rate after fine-pruning: {0}%".format(succ_att_rate))
        I_clean_remaining = np.argwhere(pred_clean == 0).ravel()
        if I_clean_remaining.size != 0:
            print('\033[0;32m' + "Please wait for accuracy...")
            test_clean_remaining = test_clean.x.take(I_clean_remaining, axis=0)
            pred_clean_remaining = np.argmax(fine_pruned_model.predict(test_clean_remaining), axis = 1)
            accu = ((test_clean.y.shape[0] - np.sum(pred_clean_remaining != test_clean.y[I_clean_remaining])) / test_clean.y.shape[0]) * 100
            print('\033[95m' + "Accuracy after fine-pruning: {0}%".format(accu))
            for i, newpred in enumerate(pred_clean_remaining):
                pred_clean[I_clean_remaining[i]] = newpred

        # Output the final prediction
        print('\033[95m' + "Final prediction of poisoned data (backdoor = -1): \n{0}".format(pred_poisoned))
        print("Final prediction of clean data (backdoor = -1): \n{0}".format(pred_clean))
    else:
        print("Mixed mode selected!\n")
        # Set the test mixed data
        print('\033[0;32m' + "Please put the mixed data under data/ and name the file mixed_data.h5 (i.e. data/mixed_data.h5) \nThen click enter.")
        input()
        mixed_data_test_filename = "data/mixex_data.h5"
        while not os.path.isfile(mixed_data_test_filename):
            print('\033[91m' + "Error: data/mixed_data.h5 does not exist. please try again.\nThen click enter.")
            input()
        test_mixed = DataLoader(mixed_data_test_filename)

        # Step one: STRIP
        print('\033[0;32m' + "STRIP: running......(maybe several minutes, even hours)")
        entropy_filename = net["entropy"]
        entropy_clean_data = h5py.File(entropy_filename, "r")
        entropy_clean = np.asarray(entropy_clean_data["data"])
        STRIP_filter = STRIP(50, 1.5, 1, 0)
        pred_mixed = STRIP_filter.predict(entropy_clean, test_mixed.x, bd_model)
        I_poisoned = np.argwhere(pred_mixed == 1).ravel()
        for i in I_poisoned:
            pred_mixed[i] = -1

        # Step two: Fine-pruning
        print('\033[0;32m' + "Fine-pruning: running......")
        I_mixed_remaining = np.argwhere(pred_mixed == 0).ravel()
        if I_mixed_remaining.size != 0:
            test_mixed_remaining = test_mixed.x.take(I_mixed_remaining, axis = 0)
            fine_pruned_model = keras.models.load_model(net["fine_pruned_model"])
            pred_mixed_remaining = fine_pruned_model.predict(test_mixed_remaining)
            for i, newpred in enumerate(pred_mixed_remaining):
                pred_mixed[I_mixed_remaining[i]] = newpred

        # Output the final prediction
        print('\033[95m' + "Final prediction of poisoned data (backdoor = -1): \n{0}".format(pred_mixed))


if __name__ == "__main__":
    main()

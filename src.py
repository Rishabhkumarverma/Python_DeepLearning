from utils.network import Net
obj = Net()

# dataset_folder_path ='/media/risverm/in_ssdv2/01_Datasets/MNIST_data'
# learning_rate = 0.001
# epoch = 50
# batch_size = 1
# obj.start_training_mnist(dataset_folder_path, batch_size, learning_rate, epoch)

dataset = dict()
dataset["data"]= '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/data.bin'
dataset["label"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/labels.bin'
dataset["test"] = '/media/risverm/in_ssdv2/01_Datasets/data_arjun_assignment/test.bin'
learning_rate = 0.001
epoch = 200
batch_size = 32
obj.start_training(dataset, batch_size, learning_rate, epoch)

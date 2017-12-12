import numpy as np
import matplotlib.pyplot as plt



def main():
    model_name = "MCB_RELU"
    history = np.genfromtxt("model_"+model_name+"_history.csv", delimiter=",", skip_header=1)
    history = history.T

    graphs = {}

    if history.shape[0] == 4:
        graphs["Accuracy"] = [history[1], history[3]]
        graphs["Loss"] = [history[0], history[2]]
    else:
        graphs["Accuracy"] = [history[1], history[4]]
        graphs["VQA Eval Accuracy"] = [history[2], history[5]]
        graphs["Loss"] = [history[0], history[3]]

    for name, data in graphs.items():
        plt.plot(data[0], marker='o', linestyle='--')
        plt.plot(data[1], marker='o', linestyle='--')
        plt.title(model_name + ' ' + name)
        plt.ylabel(name)
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(str(model_name) + "_"+name+".png")
        plt.close()


if __name__ == '__main__':
    main()
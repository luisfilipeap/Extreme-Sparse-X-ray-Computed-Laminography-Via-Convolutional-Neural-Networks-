import torch
import time
import os
from torch.autograd import Variable
import scipy.misc
import scipy
from data_loader import Tomographic_Dataset
from torch.utils.data import Dataset, DataLoader
from data_utils import data_mean_value
import numpy as np
import ntpath
from matplotlib import pyplot as plt
from torchvision import utils
#from skimage.morphology import disk
#from skimage.filters.rank import median
from imageio import imwrite


noise           =  int(1*pow(10,4))
net             = 'GoogLenet3D-ICTAI'
projs           =  4
input_dir       = "D:\\Datasets\\demo_data_plates_64_{}\\input\\".format(noise)
target_dir      = "D:\\Datasets\\demo_data_plates_64_-1\\output\\"
means           = data_mean_value("test_ictai.csv", input_dir) / 255.

model_src = "./models/{}-model".format(net)



def evaluate_img():

    test_data = Tomographic_Dataset(csv_file="test_ictai.csv", phase='val', flip_rate=0, train_csv="test_ictai.csv",
                                    input_dir=input_dir, target_dir=target_dir)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1)

    fcn_model = torch.load(model_src)
    n_tests = len(test_data.data)

    print("{} files for testing....".format(n_tests))

    folder = "./results-{}-noise-{}/".format(net, noise)
    if not os.path.exists(folder):
        os.makedirs(folder)

    execution_time = np.zeros((n_tests, 1))
    count = 0
    for iter, batch in enumerate(test_loader):

        name = batch['file'][0]
        dest = os.path.join(folder, name)
        if not os.path.exists(dest):
            os.mkdir(dest)

        if not os.path.exists(dest+"//pred"):
            os.mkdir(dest+"//pred")
        if not os.path.exists(dest + "//gt"):
            os.mkdir(dest + "//gt")
        #if not os.path.exists(dest + "//input"):
        #    os.mkdir(dest + "//input")

        #print(batch['X'].shape)
        #type(batch['X'])
        input = Variable(batch['X'].cuda())
        print(input.shape)

        start = time.time()
        output = fcn_model(input)

        output = output.data.cpu().numpy()
        N, _, z, h, w = output.shape
        y = output.reshape(N, z, h, w)

        input = batch['X'].cpu().numpy()
        input = input + means[0]
        #final_rec = input[0, 0, :, :, :] - y
        final_rec = y

        end = time.time()
        elapsed = end-start
        print(elapsed)
        execution_time[count] = elapsed
        print('execution: {} seconds'.format(elapsed))

        count = count + 1






        #grid = utils.make_grid(img_batch)
        #x = grid.numpy()[::-1].transpose((1, 2, 0))

        target = batch['Y'].cpu().numpy().reshape(N, z, h, w)
        #original = batch['o'].cpu().numpy()
        #original = scipy.misc.imread(batch['o'][0], mode='RGB')

        #gt = input[0,:,:,:]-target

        for i in range(16):
            imwrite(dest+"\\pred\\slice_{:02d}.png".format(i), final_rec[0,i,:,:])
            imwrite(dest+"\\gt\\slice_{:02d}.png".format(i), target[0,i,:,:] )
            #imwrite(dest+"\\input\\slice_{:02d}.png".format(i), input[0, 0,i,:,:])
            #imwrite(dest+"\\output_silce_{}.png".format(i), y[0,i,:,:])


        #final_rec = np.transpose(final_rec)
        #scipy.misc.imsave(dest+'\\target-residual.png', target[0,:,:])
        #scipy.misc.imsave(dest+'\\residual.png', y)
        #scipy.misc.imsave(dest+'\\final_rec.png', final_rec)
        #scipy.misc.imsave(dest+'\\input.png', x)
        #scipy.misc.imsave(dest+'\\original.png', original)









        #print("executed {} of {}\n".format(iter,len(test_loader)))

    #print("mean: {}".format(np.mean(execution_time[1:n_tests])))
    #print("std: {}".format(np.std(execution_time[1:n_tests])))



if __name__ == "__main__":
    evaluate_img()


import numpy as np
import os
import time
import torch
from torch.autograd import Variable
from torchvision import transforms
from utils.misc import check_mkdir
from model.prompt_features import convnext_fea
from utils.dataset_rgb_strategy2 import test_get_loader
import cv2
import ttach as tta
from sklearn.cluster import KMeans
torch.manual_seed(2018)
torch.cuda.set_device(0)
to_pil = transforms.ToPILImage()




def Resize(image,H, W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    return image



img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
        # tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
    ]
)



def genearte_prompt_features(image_shadow_root,ckpt_path,task_list):
    t0 = time.time()
    net = convnext_fea().cuda()
    net = torch.nn.DataParallel(net)
    net.eval()
    with torch.no_grad():
        test_loader = test_get_loader(image_shadow_root, batchsize=64, trainsize=384)
        for i, (img, gt, img_name, w_, h_) in enumerate(test_loader):
            img = Variable(img)
            img_var = img.cuda()
            n, c, h, w = img_var.size()
            assert not torch.isnan(img_var).any()
            model_output = net(img_var)

            check_mkdir(os.path.join(ckpt_path, task_list))
            for j in range(n):
                # print(model_output.shape)
                result = model_output[j,:,:,:].unsqueeze(0)
                res = result.data.cpu().numpy()
                # assert np.any(np.isnan(res))
                print(i*n+j)
                np.save(os.path.join(ckpt_path,task_list, img_name[j][:-4] + '.npy'), res)
                # cv2.imwrite(os.path.join(ckpt_path, exp_name ,args['snapshot']+'epoch',task,name,task_list[3], img_name[j][:-4] + '.png'), res)


class KMEANS:
    def __init__(self, n_clusters=64, max_iter=None, verbose=True,device = torch.device("cpu")):
        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        self.centers = x[init_row]
        assert not torch.isnan(self.centers).any()

        while True:
            print("Iteration: ", self.count)
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1
        print(self.dists.shape)
        print(self.dists)
        return self.representative_sample()


    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            assert not torch.isnan(sample).any()
            assert not torch.isnan(self.centers).any()
            # dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(sample.shape, self.centers.shape) # torch.Size([1024]) torch.Size([64, 1024])
            dist = (sample - self.centers).pow(2).sum(dim=1)
            labels[i] = torch.argmin(dist)
            assert not torch.isnan(dist).any()
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))


def time_clock(matrix,device):
    a = time.time()
    k = KMEANS(max_iter=10,verbose=False,device=device)
    k.fit(matrix)
    b = time.time()
    return (b-a)/k.count

def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def kemans_choose_prompt(ckpt_path,task_list):
    device = choose_device(True)
    # gpu_speeds = []
    root1 = os.path.join(ckpt_path, task_list)
    img_list = [os.path.splitext(f) for f in os.listdir(root1)]

    for idx, img_name in enumerate(img_list):
        print(img_name)
        fea = np.load(os.path.join(ckpt_path,task_list, img_name[0] + '.npy'))
        fea = (fea.mean(axis=(2,3)))
        fea = fea/np.linalg.norm(fea,axis=1,keepdims=True)
        assert not np.any(np.isnan(fea))
        # print(fea.max(),fea.min())
        if idx==0:
            fea_total = fea
        else:
            fea_total = np.concatenate((fea_total,fea),0)
        # print(fea_total.shape)
    print(fea_total.shape)

    k = KMeans(n_clusters=64)
    dis = k.fit_transform(fea_total)
    prompt_index = np.argmin(dis,axis=0).tolist()
    prompt_list = []
    for i in prompt_index:
        prompt_list.append(img_list[i])
    print(prompt_list)


if __name__ == '__main__':
    ckpt_path = '/root/autodl-tmp/coding/DBS_code/prompt_features'
    image_sod_root = "/root/autodl-tmp/datasets/dbs/DUTS/DUTS-TR"
    image_cod_root = "/root/autodl-tmp/datasets/dbs/cod/TrainDataset/TrainDataset"
    image_shadow_root = "/root/autodl-tmp/datasets/dbs/shadow_detection/SBU/SBU-shadow/SBUTrain4KRecoveredSmall"
    image_transparent_root = "/root/autodl-tmp/datasets/dbs/transparent/train"
    image_polyp_root = "/root/autodl-tmp/datasets/dbs/polyp/TrainDataset/TrainDataset"
    image_covid_root = "/root/autodl-tmp/datasets/dbs/COVID-19_Lung_Infection_train"
    image_breast_root = "/root/autodl-tmp/datasets/dbs/breast"  
    image_skin_root = "/root/autodl-tmp/datasets/dbs/isic2018/train"  
    task_list = ['SOD', 'COD', 'Shadow', 'Transparent', 'Polyp', 'COVID', 'Breast', 'Skin']

    ##############first run the genearte_prompt_features(), then run the kemans_choose_prompt
    # genearte_prompt_features(image_breast_root,ckpt_path,task_list[6])
    kemans_choose_prompt(ckpt_path,task_list[6])

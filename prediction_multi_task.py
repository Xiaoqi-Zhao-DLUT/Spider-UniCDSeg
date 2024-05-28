import numpy as np
import os
import time
import torch
from torch.autograd import Variable
from torchvision import transforms
from utils.config import polypfive,isic2018, duts,COD10K,SBU,trans10k_easy,trans10k_hard,covid,breast
from utils.misc import check_mkdir
from model.DBS_group_prompt import FPN_group_filter_v2_one_layer_foreground_background_filter_simple_transformer_convnext_fast_prompt_infer
from utils.dataset_rgb_strategy2 import test_get_loader,image_prompt_get_loader_kmenas_choose
import torch.nn.functional as F
import cv2
import ttach as tta
torch.manual_seed(2018)
torch.cuda.set_device(0)

ckpt_path = '/root/autodl-tmp/'
exp_name = 'coding/DBS_code/saved_model'
args = {
    'snapshot': 'train_all_8task_datasets_FPN_group_filter_v2_one_layer_foreground_background_filter_simple_transformer_swin_large_384batch16_2gpu_singletask_batch4Model_50_gen',
    'crf_refine': False,
    'save_results': True
}


to_pil = transforms.ToPILImage()
image_sod_root = "/root/autodl-tmp/datasets/dbs/DUTS/DUTS-TR/DUTS-TR-Image"
image_cod_root = "/root/autodl-tmp/datasets/dbs/cod/TrainDataset/TrainDataset/Imgs"
image_shadow_root = "/root/autodl-tmp/datasets/dbs/shadow_detection/SBU/SBU-shadow/SBUTrain4KRecoveredSmall/images"
image_transparent_root = "/root/autodl-tmp/datasets/dbs/transparent/train/images"
image_polyp_root = "/root/autodl-tmp/datasets/dbs/polyp/TrainDataset/TrainDataset/image"   
image_covid_root = "/root/autodl-tmp/datasets/dbs/COVID-19_Lung_Infection_train/images"
image_breast_root = "/root/autodl-tmp/datasets/dbs/breast/images"
image_skin_root = "/root/autodl-tmp/datasets/dbs/isic2018/train/images"  


gt_sod_root = "/root/autodl-tmp/datasets/dbs/DUTS/DUTS-TR/DUTS-TR-Mask"
gt_cod_root = "/root/autodl-tmp/datasets/dbs/cod/TrainDataset/TrainDataset/GT"
gt_shadow_root = "/root/autodl-tmp/datasets/dbs/shadow_detection/SBU/SBU-shadow/SBUTrain4KRecoveredSmall/labels"
gt_transparent_root = "/root/autodl-tmp/datasets/dbs/transparent/train/mask_binary"  
gt_polyp_root = "/root/autodl-tmp/datasets/dbs/polyp/TrainDataset/TrainDataset/mask"  
gt_covid_root = "/root/autodl-tmp/datasets/dbs/COVID-19_Lung_Infection_train/masks"
gt_breast_root = "/root/autodl-tmp/datasets/dbs/breast/masks"
gt_skin_root = "/root/autodl-tmp/datasets/dbs/isic2018/train/masks"  


breast_prompt = [('benign (397)', '.npy'), ('malignant (5)', '.npy'), ('malignant (27)', '.npy'), ('benign (367)', '.npy'), ('benign (300)', '.npy'), ('benign (95)', '.npy'), ('benign (106)', '.npy'), ('benign (58)', '.npy'), ('benign (304)', '.npy'), ('benign (154)', '.npy'), ('benign (384)', '.npy'), ('benign (263)', '.npy'), ('benign (108)', '.npy'), ('malignant (58)', '.npy'), ('malignant (155)', '.npy'), ('benign (178)', '.npy'), ('benign (403)', '.npy'), ('benign (86)', '.npy'), ('benign (157)', '.npy'), ('benign (276)', '.npy'), ('benign (21)', '.npy'), ('benign (129)', '.npy'), ('benign (352)', '.npy'), ('benign (373)', '.npy'), ('benign (273)', '.npy'), ('benign (160)', '.npy'), ('malignant (129)', '.npy'), ('benign (98)', '.npy'), ('benign (354)', '.npy'), ('benign (398)', '.npy'), ('malignant (8)', '.npy'), ('malignant (151)', '.npy'), ('benign (124)', '.npy'), ('benign (411)', '.npy'), ('malignant (29)', '.npy'), ('benign (152)', '.npy'), ('benign (319)', '.npy'), ('benign (312)', '.npy'), ('malignant (170)', '.npy'), ('malignant (66)', '.npy'), ('benign (212)', '.npy'), ('benign (225)', '.npy'), ('benign (138)', '.npy'), ('benign (105)', '.npy'), ('benign (254)', '.npy'), ('benign (266)', '.npy'), ('malignant (159)', '.npy'), ('benign (151)', '.npy'), ('benign (132)', '.npy'), ('malignant (20)', '.npy'), ('malignant (39)', '.npy'), ('benign (326)', '.npy'), ('malignant (64)', '.npy'), ('benign (85)', '.npy'), ('benign (308)', '.npy'), ('benign (1)', '.npy'), ('benign (50)', '.npy'), ('benign (323)', '.npy'), ('benign (202)', '.npy'), ('malignant (168)', '.npy'), ('malignant (116)', '.npy'), ('benign (336)', '.npy'), ('benign (425)', '.npy'), ('benign (134)', '.npy')]

covid_prompt = [('009_210', '.npy'), ('007_135', '.npy'), ('11_102', '.npy'), ('009_3', '.npy'), ('36_15', '.npy'), ('004_231', '.npy'), ('7_12', '.npy'), ('11_12', '.npy'), ('007_225', '.npy'), ('008_138', '.npy'), ('010_111', '.npy'), ('008_216', '.npy'), ('008_279', '.npy'), ('11_408', '.npy'), ('005_39', '.npy'), ('003_180', '.npy'), ('001_291', '.npy'), ('27_51', '.npy'), ('004_222', '.npy'), ('40_39', '.npy'), ('007_168', '.npy'), ('003_39', '.npy'), ('11_333', '.npy'), ('11_84', '.npy'), ('004_51', '.npy'), ('77_tiny', '.npy'), ('40_15', '.npy'), ('11_153', '.npy'), ('009_99', '.npy'), ('004_189', '.npy'), ('009_240', '.npy'), ('010_57', '.npy'), ('010_291', '.npy'), ('007_237', '.npy'), ('001_15', '.npy'), ('010_183', '.npy'), ('10_0', '.npy'), ('88_tiny', '.npy'), ('14_105', '.npy'), ('004_252', '.npy'), ('006_18', '.npy'), ('003_147', '.npy'), ('005_102', '.npy'), ('002_150', '.npy'), ('11_357', '.npy'), ('010_237', '.npy'), ('002_36', '.npy'), ('10_27', '.npy'), ('14_21', '.npy'), ('14_57', '.npy'), ('001_255', '.npy'), ('006_93', '.npy'), ('11_63', '.npy'), ('46_tiny', '.npy'), ('11_300', '.npy'), ('008_264', '.npy'), ('006_150', '.npy'), ('11_243', '.npy'), ('010_9', '.npy'), ('007_30', '.npy'), ('39_tiny', '.npy'), ('40_69', '.npy'), ('005_129', '.npy'), ('009_252', '.npy')]

polyp_prompt = [('315', '.npy'), ('cju33x0f22peh0988g0ln7w5v', '.npy'), ('280', '.npy'), ('cju8dm2cau2km0818jsv9eeq2', '.npy'), ('cju15jr8jz8sb0855ukmkswkz', '.npy'), ('238', '.npy'), ('cju3v0fl3gwce0755qkjhzmd4', '.npy'), ('535', '.npy'), ('94', '.npy'), ('579', '.npy'), ('444', '.npy'), ('cju2i6acqvo6l0799u20fift8', '.npy'), ('253', '.npy'), ('189', '.npy'), ('159', '.npy'), ('cju5x15djm7ae0755h8czf6nt', '.npy'), ('cju0t4oil7vzk099370nun5h9', '.npy'), ('150', '.npy'), ('58', '.npy'), ('cju41nz76lcxu0755cya2qefx', '.npy'), ('369', '.npy'), ('586', '.npy'), ('3', '.npy'), ('510', '.npy'), ('354', '.npy'), ('38', '.npy'), ('416', '.npy'), ('cju45jpvfn6c809873pv1i34s', '.npy'), ('500', '.npy'), ('cju171py4qiha0835u8sl59ds', '.npy'), ('cju2z2x3nvd3c099350zgty7w', '.npy'), ('113', '.npy'), ('cju1dfeupuzlw0835gnxip369', '.npy'), ('334', '.npy'), ('202', '.npy'), ('cju83qd0yjyht0817ktkfl268', '.npy'), ('cju773hsyyosz0817pk1e7sjq', '.npy'), ('cju5bhv81abur0850ean02atv', '.npy'), ('cju5thdbrjp1108715xdfx356', '.npy'), ('cju87ox0kncom0801b98hqnd2', '.npy'), ('cju5f26ebcuai0818xlwh6116', '.npy'), ('45', '.npy'), ('547', '.npy'), ('cju2nbdpmlmcj0993s1cht0dz', '.npy'), ('592', '.npy'), ('476', '.npy'), ('cju5i5oh2efg60987ez6cpf72', '.npy'), ('602', '.npy'), ('449', '.npy'), ('cju5ht88gedbu0755xrcuddcx', '.npy'), ('cju1fm3id6gl50801r3fok20c', '.npy'), ('cjyzufihqquiw0a46jatrbwln', '.npy'), ('cju5uzmaol56l0817flxh4w9p', '.npy'), ('435', '.npy'), ('102', '.npy'), ('402', '.npy'), ('254', '.npy'), ('525', '.npy'), ('356', '.npy'), ('298', '.npy'), ('cju5fydrud94708507vo6oy21', '.npy'), ('cju2yb31a8e8u0878wdashg7o', '.npy'), ('183', '.npy'), ('220', '.npy')]

skin_prompt = [('1499', '.npy'), ('345', '.npy'), ('1575', '.npy'), ('903', '.npy'), ('1812', '.npy'), ('444', '.npy'), ('577', '.npy'), ('317', '.npy'), ('342', '.npy'), ('1767', '.npy'), ('1388', '.npy'), ('1444', '.npy'), ('781', '.npy'), ('1335', '.npy'), ('759', '.npy'), ('76', '.npy'), ('729', '.npy'), ('1147', '.npy'), ('1126', '.npy'), ('557', '.npy'), ('818', '.npy'), ('1667', '.npy'), ('484', '.npy'), ('1188', '.npy'), ('1141', '.npy'), ('1210', '.npy'), ('1471', '.npy'), ('1447', '.npy'), ('300', '.npy'), ('1300', '.npy'), ('1453', '.npy'), ('295', '.npy'), ('1475', '.npy'), ('964', '.npy'), ('521', '.npy'), ('316', '.npy'), ('855', '.npy'), ('946', '.npy'), ('1454', '.npy'), ('1481', '.npy'), ('1206', '.npy'), ('517', '.npy'), ('370', '.npy'), ('1730', '.npy'), ('487', '.npy'), ('1830', '.npy'), ('100', '.npy'), ('1417', '.npy'), ('1784', '.npy'), ('872', '.npy'), ('1528', '.npy'), ('1773', '.npy'), ('1098', '.npy'), ('949', '.npy'), ('687', '.npy'), ('158', '.npy'), ('330', '.npy'), ('1237', '.npy'), ('628', '.npy'), ('931', '.npy'), ('604', '.npy'), ('954', '.npy'), ('641', '.npy'), ('419', '.npy')]

transparent_prompt = [('8631', '.npy'), ('2256', '.npy'), ('5489', '.npy'), ('10222', '.npy'), ('3780', '.npy'), ('121', '.npy'), ('8531', '.npy'), ('7141', '.npy'), ('7603', '.npy'), ('10326', '.npy'), ('4613', '.npy'), ('5988', '.npy'), ('1443', '.npy'), ('3145', '.npy'), ('4352', '.npy'), ('9153', '.npy'), ('9189', '.npy'), ('4828', '.npy'), ('5875', '.npy'), ('4326', '.npy'), ('5853', '.npy'), ('6426', '.npy'), ('5157', '.npy'), ('5385', '.npy'), ('8867', '.npy'), ('6770', '.npy'), ('2857', '.npy'), ('4389', '.npy'), ('10204', '.npy'), ('3787', '.npy'), ('9874', '.npy'), ('4339', '.npy'), ('1372', '.npy'), ('9391', '.npy'), ('280', '.npy'), ('4280', '.npy'), ('10287', '.npy'), ('1437', '.npy'), ('5575', '.npy'), ('9537', '.npy'), ('6299', '.npy'), ('9908', '.npy'), ('1846', '.npy'), ('518', '.npy'), ('5807', '.npy'), ('2171', '.npy'), ('3115', '.npy'), ('1324', '.npy'), ('4199', '.npy'), ('1595', '.npy'), ('6390', '.npy'), ('8455', '.npy'), ('5221', '.npy'), ('2083', '.npy'), ('806', '.npy'), ('868', '.npy'), ('10355', '.npy'), ('705', '.npy'), ('1375', '.npy'), ('2245', '.npy'), ('5073', '.npy'), ('6239', '.npy'), ('2018', '.npy'), ('7214', '.npy')]

shadow_prompt = [('lssd4036', '.npy'), ('lssd3189', '.npy'), ('lssd1810', '.npy'), ('lssd3847', '.npy'), ('lssd4100', '.npy'), ('lssd2654', '.npy'), ('lssd1697', '.npy'), ('lssd1521', '.npy'), ('lssd1970', '.npy'), ('lssd2264', '.npy'), ('lssd2659', '.npy'), ('lssd3904', '.npy'), ('lssd3356', '.npy'), ('lssd2390', '.npy'), ('lssd2074', '.npy'), ('lssd2723', '.npy'), ('lssd2239', '.npy'), ('lssd3666', '.npy'), ('lssd1120', '.npy'), ('lssd2835', '.npy'), ('lssd3846', '.npy'), ('lssd73', '.npy'), ('lssd2704', '.npy'), ('lssd1939', '.npy'), ('lssd1126', '.npy'), ('lssd3290', '.npy'), ('lssd1165', '.npy'), ('lssd1285', '.npy'), ('lssd1598', '.npy'), ('lssd787', '.npy'), ('lssd1503', '.npy'), ('lssd3308', '.npy'), ('lssd2062', '.npy'), ('lssd535', '.npy'), ('lssd1168', '.npy'), ('lssd825', '.npy'), ('lssd1379', '.npy'), ('lssd2514', '.npy'), ('lssd555', '.npy'), ('lssd1817', '.npy'), ('lssd491', '.npy'), ('lssd1950', '.npy'), ('lssd3467', '.npy'), ('lssd2087', '.npy'), ('lssd808', '.npy'), ('lssd3126', '.npy'), ('lssd3941', '.npy'), ('lssd1206', '.npy'), ('lssd1229', '.npy'), ('lssd324', '.npy'), ('lssd3942', '.npy'), ('lssd3855', '.npy'), ('lssd3582', '.npy'), ('lssd3300', '.npy'), ('lssd2050', '.npy'), ('lssd287', '.npy'), ('lssd512', '.npy'), ('lssd3744', '.npy'), ('lssd1463', '.npy'), ('lssd864', '.npy'), ('lssd720', '.npy'), ('lssd1087', '.npy'), ('lssd3865', '.npy'), ('lssd983', '.npy')]

sod_prompt = [('n07615774_37850', '.npy'), ('n03710721_15203', '.npy'), ('n03271574_6388', '.npy'), ('n04146614_7148', '.npy'), ('ILSVRC2012_test_00018248', '.npy'), ('n03188531_32001', '.npy'), ('n07768694_5450', '.npy'), ('n07739125_18914', '.npy'), ('n07742313_3643', '.npy'), ('n07749582_3656', '.npy'), ('n03775546_5435', '.npy'), ('n03127747_2170', '.npy'), ('n04263257_3334', '.npy'), ('n03445777_6918', '.npy'), ('n07873807_14779', '.npy'), ('n04254680_4926', '.npy'), ('n03764736_222', '.npy'), ('n07714571_2421', '.npy'), ('n04487081_9489', '.npy'), ('n04026417_36464', '.npy'), ('n04023962_6185', '.npy'), ('n03513137_8020', '.npy'), ('n04392985_9128', '.npy'), ('n07718472_28911', '.npy'), ('n03769881_4642', '.npy'), ('ILSVRC2012_test_00045866', '.npy'), ('n03188531_32387', '.npy'), ('n03761084_9988', '.npy'), ('n04037443_19027', '.npy'), ('n03379051_6502', '.npy'), ('ILSVRC2012_test_00070024', '.npy'), ('n07720875_4650', '.npy'), ('n07753113_319', '.npy'), ('n07930864_10909', '.npy'), ('ILSVRC2012_test_00099417', '.npy'), ('n07747607_60948', '.npy'), ('ILSVRC2013_val_00003498', '.npy'), ('ILSVRC2012_val_00044972', '.npy'), ('n03594945_37686', '.npy'), ('n06874185_23178', '.npy'), ('n07695742_968', '.npy'), ('n04019541_1374', '.npy'), ('ILSVRC2012_test_00035259', '.npy'), ('n04074963_505', '.npy'), ('n03337140_34301', '.npy'), ('n04344873_9734', '.npy'), ('n07745940_20623', '.npy'), ('n01667114_3911', '.npy'), ('ILSVRC2012_val_00042701', '.npy'), ('n03584254_554', '.npy'), ('n03124170_1949', '.npy'), ('n04118538_7436', '.npy'), ('ILSVRC2014_train_00021048', '.npy'), ('ILSVRC2012_test_00078940', '.npy'), ('n07718747_4108', '.npy'), ('n01755581_4275', '.npy'), ('n03770439_18812', '.npy'), ('n03670208_4237', '.npy'), ('n02165456_441', '.npy'), ('ILSVRC2012_test_00022088', '.npy'), ('ILSVRC2012_test_00043873', '.npy'), ('n03690938_7187', '.npy'), ('n04371430_6186', '.npy'), ('n04273569_14846', '.npy')]

cod_prompt = [('COD10K-CAM-2-Terrestrial-44-Snake-2429', '.npy'), ('COD10K-CAM-2-Terrestrial-46-StickInsect-2874', '.npy'), ('COD10K-CAM-3-Flying-65-Owl-4570', '.npy'), ('COD10K-CAM-1-Aquatic-6-Fish-161', '.npy'), ('COD10K-CAM-3-Flying-62-Mantis-4299', '.npy'), ('COD10K-CAM-3-Flying-65-Owl-4625', '.npy'), ('COD10K-CAM-4-Amphibian-67-Frog-4736', '.npy'), ('COD10K-CAM-1-Aquatic-6-Fish-206', '.npy'), ('COD10K-CAM-3-Flying-59-Grasshopper-3730', '.npy'), ('COD10K-CAM-1-Aquatic-15-SeaHorse-1028', '.npy'), ('COD10K-CAM-2-Terrestrial-47-Tiger-2892', '.npy'), ('COD10K-CAM-3-Flying-55-Butterfly-3292', '.npy'), ('COD10K-CAM-1-Aquatic-3-Crab-91', '.npy'), ('COD10K-CAM-2-Terrestrial-45-Spider-2661', '.npy'), ('COD10K-CAM-1-Aquatic-20-Turtle-1218', '.npy'), ('COD10K-CAM-3-Flying-58-Frogmouth-3597', '.npy'), ('COD10K-CAM-1-Aquatic-9-GhostPipefish-408', '.npy'), ('COD10K-CAM-2-Terrestrial-38-Lizard-2302', '.npy'), ('COD10K-CAM-1-Aquatic-13-Pipefish-524', '.npy'), ('COD10K-CAM-2-Terrestrial-36-Leopard-2078', '.npy'), ('camourflage_00804', '.npy'), ('COD10K-CAM-2-Terrestrial-34-Human-2016', '.npy'), ('COD10K-CAM-1-Aquatic-7-Flounder-282', '.npy'), ('COD10K-CAM-3-Flying-62-Mantis-4334', '.npy'), ('COD10K-CAM-4-Amphibian-68-Toad-4819', '.npy'), ('COD10K-CAM-3-Flying-59-Grasshopper-3787', '.npy'), ('COD10K-CAM-3-Flying-61-Katydid-4200', '.npy'), ('camourflage_00044', '.npy'), ('COD10K-CAM-3-Flying-61-Katydid-3941', '.npy'), ('camourflage_00743', '.npy'), ('COD10K-CAM-3-Flying-53-Bird-3110', '.npy'), ('COD10K-CAM-3-Flying-61-Katydid-4066', '.npy'), ('COD10K-CAM-1-Aquatic-18-StarFish-1163', '.npy'), ('COD10K-CAM-3-Flying-63-Mockingbird-4415', '.npy'), ('COD10K-CAM-1-Aquatic-15-SeaHorse-1031', '.npy'), ('COD10K-CAM-2-Terrestrial-45-Spider-2503', '.npy'), ('COD10K-CAM-2-Terrestrial-28-Deer-1778', '.npy'), ('COD10K-CAM-2-Terrestrial-38-Lizard-2177', '.npy'), ('camourflage_01212', '.npy'), ('COD10K-CAM-2-Terrestrial-24-Caterpillar-1604', '.npy'), ('COD10K-CAM-2-Terrestrial-32-Giraffe-1946', '.npy'), ('COD10K-CAM-3-Flying-55-Butterfly-3405', '.npy'), ('COD10K-CAM-2-Terrestrial-26-Chameleon-1710', '.npy'), ('camourflage_00704', '.npy'), ('COD10K-CAM-2-Terrestrial-43-Sheep-2421', '.npy'), ('COD10K-CAM-2-Terrestrial-38-Lizard-2163', '.npy'), ('COD10K-CAM-3-Flying-53-Bird-3090', '.npy'), ('COD10K-CAM-2-Terrestrial-42-Sciuridae-2414', '.npy'), ('COD10K-CAM-4-Amphibian-68-Toad-5007', '.npy'), ('COD10K-CAM-3-Flying-54-Bittern-3257', '.npy'), ('camourflage_00437', '.npy'), ('COD10K-CAM-3-Flying-57-Dragonfly-3568', '.npy'), ('COD10K-CAM-3-Flying-51-Bee-3004', '.npy'), ('COD10K-CAM-1-Aquatic-15-SeaHorse-1089', '.npy'), ('COD10K-CAM-3-Flying-60-Heron-3923', '.npy'), ('COD10K-CAM-2-Terrestrial-23-Cat-1385', '.npy'), ('COD10K-CAM-1-Aquatic-13-Pipefish-516', '.npy'), ('COD10K-CAM-1-Aquatic-14-ScorpionFish-877', '.npy'), ('camourflage_01136', '.npy'), ('COD10K-CAM-2-Terrestrial-29-Dog-1861', '.npy'), ('camourflage_00261', '.npy'), ('camourflage_00720', '.npy'), ('COD10K-CAM-3-Flying-56-Cicada-3464', '.npy'), ('COD10K-CAM-1-Aquatic-20-Turtle-1214', '.npy')]


to_test = {'DUTS':duts} #images,gt
# to_test = {'COD10K':COD10K} # Imgs, GT
# to_test = {'SBU':SBU} # Image, Mask
# to_test = {'easy':trans10k_easy,'hard':trans10k_hard} # images, mask_binary
# to_test = {'polyp':polypfive} # images,masks
# to_test = {'COVID':covid}# images,masks
# to_test = {'Breast':breast}# images,masks
# to_test = {'ISIC2018':isic2018}# images,masks



task = 'SOD_kmeans_prompt'




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

task_list = ['SOD','COD','Shadow','Transparent','Polyp','COVID','Breast','Skin']

def main():
    t0 = time.time()
    net = FPN_group_filter_v2_one_layer_foreground_background_filter_simple_transformer_convnext_fast_prompt_infer().cuda()
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot']+'.pth'),map_location={'cuda:1': 'cuda:1'}))
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root)

            test_loader = test_get_loader(root1, batchsize=64,trainsize=384)
            train_sod_loader = image_prompt_get_loader_kmenas_choose(image_sod_root, gt_sod_root, sod_prompt, batchsize=64,
                                             trainsize=384)
            # train_cod_loader = image_prompt_get_loader_kmenas_choose(image_cod_root, gt_cod_root, cod_prompt, batchsize=64,
            #                                  trainsize=384)
            # train_shadow_loader = image_prompt_get_loader_kmenas_choose(image_shadow_root, gt_shadow_root, shadow_prompt, batchsize=64,
            #                                  trainsize=384)
            # train_transparent_loader = image_prompt_get_loader_kmenas_choose(image_transparent_root, gt_transparent_root, transparent_prompt, batchsize=64,
            #                                  trainsize=384)
            # train_polyp_loader = image_prompt_get_loader_kmenas_choose(image_polyp_root, gt_polyp_root, polyp_prompt, batchsize=64,
            #                                 trainsize=384)
            # train_covid_loader = image_prompt_get_loader_kmenas_choose(image_covid_root, gt_covid_root,covid_prompt, batchsize=64, trainsize=384)
            # train_breast_loader = image_prompt_get_loader_kmenas_choose(image_breast_root, gt_breast_root, breast_prompt, batchsize=64, trainsize=384)
            # train_skin_loader = image_prompt_get_loader_kmenas_choose(image_skin_root, gt_skin_root, skin_prompt, batchsize=64, trainsize=384)

            for i , (image_sod, gt_sod) in enumerate(train_sod_loader):
                image_sod = Variable(image_sod)
                gt_sod = Variable(gt_sod)
                image_sod_var = image_sod.cuda()
                gt_sod_var = gt_sod.cuda()

                filter_list = [image_sod_var]
                gt_list = [gt_sod_var]
                prompt_conv = net(x=None, filter_list=filter_list, mask_list = gt_list,generate_filter=True,prompt_kernel=None,backbone_name='swin')
                for i,  (img, gt, img_name, w_, h_) in enumerate(test_loader):
                    img = Variable(img)
                    gt = Variable(gt)
                    img_var = img.cuda()
                    n, c, h, w = img_var.size()

                    model_output = net(x=img_var, filter_list=None, mask_list=None,generate_filter=False,prompt_kernel=prompt_conv,backbone_name='swin')
                    prediction = model_output.sigmoid()
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'SOD'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'COD'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'Shadow'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'Transparent'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'Polyp'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'COVID'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'Breast'))
                    check_mkdir(os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, 'Skin'))
                    for j in range(n):
                        for k in range(len(filter_list)):
                            print(prediction.shape)
                            result = prediction[j, k:k + 1, :, :].unsqueeze(0)
                            res = F.upsample(result, size=[h_[j], w_[j]], mode='bilinear', align_corners=False)
                            res = res.data.cpu().numpy().squeeze()
                            res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)
                            print(img_name[j])
                            cv2.imwrite(
                                os.path.join(ckpt_path, exp_name, args['snapshot'] + 'epoch', task, name, task_list[0],
                                             img_name[j][:-4] + '.png'), res)

if __name__ == '__main__':
    main()

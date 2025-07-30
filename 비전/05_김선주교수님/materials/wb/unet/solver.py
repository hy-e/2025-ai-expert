import torch,os,random,time,rawpy,json

from tqdm import tqdm
from torch import optim
from torch import nn
from model import U_Net
from utils import *
from modules import *
from metrics import *
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

class Solver():
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # RAW args
        self.camera = config.camera
        self.raw = rawpy.imread(self.camera+".dng")
        self.white_level = self.raw.white_level
        if self.camera == 'sony':
            self.white_level = self.white_level/4

        # Training config
        self.mode = config.mode
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.cl_coeff = config.cl_coeff
        self.criterion = nn.MSELoss(reduction='mean')
        #self.CLLoss = ColorLineLoss()
        self.num_epochs_decay = config.num_epochs_decay
        self.save_epoch = config.save_epoch

        # Data loader
        self.data_root = config.data_root
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.input_type = config.input_type
        self.output_type = config.output_type

        # Models
        self.net = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.checkpoint = config.checkpoint
        self.pre_checkpoint = config.pre_checkpoint
        self.abstract_pool_size = config.abstract_pool_size
        self.pre_abstract_pool_size = config.pre_abstract_pool_size

        # Visualize step
        self.save_result = config.save_result
        self.vis_step = config.vis_step
        self.ae_map_dir = config.ae_map_dir
        # Misc
        if config.checkpoint:
            self.train_date = self.checkpoint.split('/')[0] # get base directory from checkpoint
        else:
            self.train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        if config.pre_checkpoint:
            self.pre_train_date = self.pre_checkpoint.split('/')[0] # get base directory from checkpoint
        else:
            self.pre_train_date = time.strftime("%y%m%d_%H%M", time.localtime(time.time()))
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.log_interval = config.log_interval

        # Initialize default path & SummaryWriter
        self.model_path = os.path.join(config.model_root,self.train_date)
        self.pre_model_path = os.path.join(config.pre_model_root,self.pre_train_date)
        self.result_path = os.path.join(config.result_root,self.train_date+'_'+self.mode)
        if os.path.isdir(self.model_path) == False:
            os.makedirs(self.model_path)
        if os.path.isdir(self.result_path) == False and self.save_result == 'yes':
            os.makedirs(self.result_path)
        if self.mode == "train":
            self.log_path = os.path.join(config.log_root,self.train_date)
            # self.writer = SummaryWriter(self.log_path)
            with open(os.path.join(self.model_path,'args.txt'), 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            f.close()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.build_model()

    def build_model(self):
        # build network, configure optimizer
        print("[Model]\tBuilding Network...")

        if self.pre_checkpoint != None:
            self.pre_net = U_Net(img_ch=self.img_ch, output_ch=self.output_ch, abstract_pool_size=self.pre_abstract_pool_size)
            self.net = U_Net(img_ch=self.img_ch+self.output_ch, output_ch=self.output_ch, abstract_pool_size=self.abstract_pool_size)
        else:
            self.net = U_Net(img_ch=self.img_ch, output_ch=self.output_ch, abstract_pool_size=self.abstract_pool_size)

        # Load model from checkpoint
        if self.checkpoint != None:
            ckpt_file = 'best.pt' if '/' not in self.checkpoint else os.path.split(self.checkpoint)[1]
            ckpt = os.path.join(self.model_path,ckpt_file)
            print("[Model]\tLoad model from checkpoint :", ckpt)
            self.net.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))

        # Load pre_model from checkpoint
        if self.pre_checkpoint != None:
            pre_ckpt_file = 'best.pt' if '/' not in self.pre_checkpoint else os.path.split(self.pre_checkpoint)[1]
            pre_ckpt = os.path.join(self.pre_model_path,pre_ckpt_file)
            print("[Model]\tLoad pre_model from pre_checkpoint :", pre_ckpt)
            self.pre_net.load_state_dict(torch.load(pre_ckpt))

        # multi-GPU
        if torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net)
            if self.pre_checkpoint != None:
                self.pre_net = nn.DataParallel(self.pre_net)
        
        # GPU & optimizer
        self.net.to(self.device)
        if self.pre_checkpoint != None:
            self.pre_net.to(self.device)
        self.optimizer = optim.Adam(list(self.net.parameters()),
                                      self.lr, [self.beta1, self.beta2])
        print("[Model]\tBuild Complete.")

    def test(self):
        print("[Test]\tStart testing process.")
        self.net.eval()

        test_loss = []
        test_pred_loss = []
        test_cl_loss = []
        test_MAE_illum = []
        test_MAE_illum_abstract = []
        test_MAE_rgb = []
        test_PSNR = []

        for i, batch in enumerate(tqdm(self.test_loader)):
            # prepare input,GT,mask
            if self.input_type == "rgb":
                inputs = batch["input_rgb"].to(self.device)
            elif self.input_type == "uvl":
                inputs = batch["input_uvl"].to(self.device)
            if self.output_type == "illumination":
                if self.abstract_pool_size > 1:
                    GTs = get_abstract_illum_map(batch["gt_illum"], self.abstract_pool_size)
                    GTs = torch.log(GTs + 1e-8).to(self.device)
                else:
                    GTs = torch.log(batch["gt_illum"] + 1e-8).to(self.device)
            elif self.output_type == "uv":
                GTs = batch["gt_uv"].to(self.device)
            masks = batch["mask"].to(self.device)
            if self.abstract_pool_size > 1:
                masks_abstract = get_abstract_illum_map(batch["mask"], self.abstract_pool_size)
                masks_abstract = (masks_abstract > 0.5).float().to(self.device)

            # [JW]
            if self.pre_checkpoint != None:
                self.pre_net.eval()
                # prepare gt_illum
                gt_illum_uv = torch.log(batch['gt_illum'] + 1e-8).to(self.device)

                # pred pre_illum and train
                pre_pred_illum = self.pre_net(inputs)

                if self.pre_abstract_pool_size > 1:
                    pre_pred_illum = nn.Upsample(scale_factor=self.pre_abstract_pool_size, mode='bilinear', align_corners=True)(pre_pred_illum)

                # cat input with pre_pred_illum
                inputs = torch.cat([inputs, pre_pred_illum.to(self.device)], axis=1)

            # inference
            pred = self.net(inputs)
            pred_detach = pred.detach()
            pred_loss = self.criterion(pred, GTs)

            # linear colorline loss
            if self.output_type == "illumination":
                if self.abstract_pool_size > 1:
                    illum_map_rb = torch.exp(pred * masks_abstract)
                else:
                    illum_map_rb = torch.exp(pred * masks)
            elif self.output_type == "uv":
                # difference of uv (inputs-pred) equals to illumination RB value
                illum_map_rb = torch.exp((inputs[:,0:2,:,:]-pred)*masks)
            #cl_loss = self.CLLoss(illum_map_rb)
            
            total_loss = pred_loss #+ self.cl_coeff*cl_loss
            
            # calculate pred_rgb & pred_illum & gt_illum
            input_rgb = batch["input_rgb"].to(self.device)
            gt_illum = batch["gt_illum"].to(self.device)
            if self.abstract_pool_size > 1:
                gt_illum_abstract = get_abstract_illum_map(gt_illum, self.abstract_pool_size)
            gt_rgb = batch["gt_rgb"].to(self.device)

            if self.output_type == "illumination":
                ones = torch.ones_like(pred_detach[:,:1,:,:])
                pred_illum = torch.cat([pred_detach[:,:1,:,:],ones,pred_detach[:,1:,:,:]],dim=1)
                # take exp since the pred_illum is in log scale
                pred_illum = torch.exp(pred_illum)
                if self.abstract_pool_size > 1:
                    pred_illum_abstract = pred_illum
                    pred_illum = nn.Upsample(scale_factor=self.abstract_pool_size, mode='bilinear', align_corners=True)(pred_illum_abstract)
                    pred_illum_abstract[:,1,:,:] = 1.
                pred_illum[:,1,:,:] = 1.
                pred_rgb = apply_wb(input_rgb,pred_illum,pred_type='illumination')
            elif self.output_type == "uv":
                pred_rgb = apply_wb(input_rgb,pred_detach,pred_type='uv')
                pred_illum = input_rgb / (pred_rgb + 1e-8)
            ones = torch.ones_like(gt_illum[:,:1,:,:])
            gt_illum = torch.cat([gt_illum[:,:1,:,:],ones,gt_illum[:,1:,:,:]],dim=1)
            if self.abstract_pool_size > 1:
                ones = torch.ones_like(gt_illum_abstract[:,:1,:,:])
                gt_illum_abstract = torch.cat([gt_illum_abstract[:,:1,:,:],ones,gt_illum_abstract[:,1:,:,:]],dim=1)

            # error metrics
            MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"])
            if self.abstract_pool_size > 1:
                MAE_illum_abstract = get_MAE(pred_illum_abstract,gt_illum_abstract,tensor_type="illumination",mask=masks_abstract)
            else:
                MAE_illum_abstract = torch.Tensor([-1.])
            MAE_rgb = get_MAE(pred_rgb,gt_rgb,tensor_type="rgb",camera=self.camera,mask=batch["mask"])
            PSNR = get_PSNR(pred_rgb,gt_rgb,white_level=self.white_level)

            # draw and save pixel-level mae map
            # [B(=1), H, W]
            pixel_level_MAE_illum = get_MAE(pred_illum,gt_illum,tensor_type="illumination",mask=batch["mask"],pixel_level=True)
            draw_AE_map(pixel_level_MAE_illum[0], fname=batch['img_file'][0][:-5], mae=MAE_illum, psnr=PSNR, ae_map_dir=self.ae_map_dir)

            print(f'[Test] Batch [{i+1}/{len(self.test_loader)}] | ' \
                        f'total_loss:{total_loss.item():.5f} | ' \
                        f'pred_loss:{pred_loss.item():.5f} | ' \
                        #f'cl_loss:{cl_loss.item():.5f} | ' \
                        f'MAE_illum:{MAE_illum.item():.5f} | '\
                        f'MAE_illum_abstract:{MAE_illum_abstract.item():.5f} | '\
                        f'MAE_rgb:{MAE_rgb.item():.5f} | '\
                        f'PSNR:{PSNR.item():.5f}')

            test_loss.append(total_loss.item())
            test_pred_loss.append(pred_loss.item())
            #test_cl_loss.append(cl_loss.item())
            test_MAE_illum.append(MAE_illum.item())
            test_MAE_illum_abstract.append(MAE_illum_abstract.item())
            test_MAE_rgb.append(MAE_rgb.item())
            test_PSNR.append(PSNR.item())

            if self.save_result == 'yes':
                # plot illumination map to R,B space
                plot_fig = plot_illum(pred_map=illum_map_rb.permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 gt_map=gt_illum[:,[0,2],:,:].permute(0,2,3,1).reshape((-1,2)).cpu().detach().numpy(),
                                 MAE_illum=MAE_illum,MAE_rgb=MAE_rgb,PSNR=PSNR)
                srgb_visualized = visualize(batch['input_rgb'][0],pred_rgb[0],batch['gt_rgb'][0],self.camera,concat=True)

                cv2.imwrite(os.path.join(self.result_path,batch["place"][0]+'_'+batch["illum_count"][0]+'_plot.png'),plot_fig)
                cv2.imwrite(os.path.join(self.result_path,batch["place"][0]+'_'+batch["illum_count"][0]+'_vis.png'),cv2.cvtColor(srgb_visualized,cv2.COLOR_RGB2BGR))

        print("loss :", np.mean(test_loss), np.median(test_loss), np.max(test_loss))
        print("pred_loss :", np.mean(test_pred_loss), np.median(test_pred_loss), np.max(test_pred_loss))
        #print("cl_loss :", np.mean(test_cl_loss), np.median(test_cl_loss), np.max(test_cl_loss))
        print("MAE_illum :", np.mean(test_MAE_illum), np.median(test_MAE_illum), np.max(test_MAE_illum))
        print("MAE_illum_abstract :", np.mean(test_MAE_illum_abstract), np.median(test_MAE_illum_abstract), np.max(test_MAE_illum_abstract))
        print("MAE_rgb :", np.mean(test_MAE_rgb), np.median(test_MAE_rgb), np.max(test_MAE_rgb))
        print("PSNR :", np.mean(test_PSNR), np.median(test_PSNR), np.max(test_PSNR))

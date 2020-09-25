#####################################################################
# Import Packages
#####################################################################

import numpy as np
import torch
import os
from .base_model import BaseModel
from models.networks import Define_G, Define_D
from utils.transforms import create_part
import torch.nn.functional as F
from utils import pose_utils
from lib.geometric_matching_multi_gpu import GMM
from .base_model import BaseModel
from time import time
from utils import pose_utils
from utils.warp_image import warped_image
import os.path as osp
from torchvision import utils
import random
from torch.utils.tensorboard import SummaryWriter

#####################################################################
# DNN Functions
#####################################################################


class GenerationModel(BaseModel):
    # Returns the name of the Network
    def name(self):
        return 'Generation model: pix2pix | pix2pixHD'

    # Init Function
    def __init__(self, opt):
        self.t0 = time()

        BaseModel.__init__(self, opt)
        self.train_mode = opt.train_mode

        # Resume of networks
        resume_gmm = opt.resume_gmm
        resume_G_parse = opt.resume_G_parse
        resume_D_parse = opt.resume_D_parse
        resume_G_appearance = opt.resume_G_app
        resume_D_appearance = opt.resume_D_app
        resume_G_face = opt.resume_G_face
        resume_D_face = opt.resume_D_face

        # Define network
        self.gmm_model = torch.nn.DataParallel(GMM(opt)).cuda()
        self.generator_parsing = Define_G(opt.input_nc_G_parsing, opt.output_nc_parsing, opt.ndf, opt.netG_parsing, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.discriminator_parsing = Define_D(opt.input_nc_D_parsing, opt.ndf, opt.netD_parsing, opt.n_layers_D,
                                              opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.generator_appearance = Define_G(opt.input_nc_G_app, opt.output_nc_app, opt.ndf, opt.netG_app, opt.norm,
                                             not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids, with_tanh=False)
        self.discriminator_appearance = Define_D(opt.input_nc_D_app, opt.ndf, opt.netD_app, opt.n_layers_D,
                                                 opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.generator_face = Define_G(opt.input_nc_D_face, opt.output_nc_face, opt.ndf, opt.netG_face, opt.norm,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.discriminator_face = Define_D(opt.input_nc_D_face, opt.ndf, opt.netD_face, opt.n_layers_D,
                                           opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)

        ######################################
        # HELPER COMMENTS
        # when we train each network seperately we use generator and discriminator according to train_mode
        # when we train jointly we train appearance network generator and discriminator
        # when we train jointly parsing/ face generator network gets trained only
        # self.generator/discriminator optimizer also follows the same strategy and  comes from train mode variable
        ######################################
        print("Train Mode is --- ", opt.train_mode)

        if opt.train_mode == 'gmm':
            setattr(self, 'generator', self.gmm_model)
        else:
            setattr(self, 'generator', getattr(
                self, 'generator_' + self.train_mode))
            setattr(self, 'discriminator', getattr(
                self, 'discriminator_' + self.train_mode))

        ######################################
        # Load Networks
        ######################################
        self.networks_name = ['gmm', 'G_parsing',
                              'D_parsing', 'G_appearance', 'D_appearance', 'G_face', 'D_face']
        self.networks_model = [
            self.gmm_model, self.generator_parsing, self.discriminator_parsing, self.generator_appearance, self.discriminator_appearance, self.generator_face, self.discriminator_face]
        self.networks = dict(zip(self.networks_name, self.networks_model))
        self.resume_path = [resume_gmm, resume_G_parse, resume_D_parse,
                            resume_G_appearance, resume_D_appearance, resume_G_face, resume_D_face]

        for index, (network, resume) in enumerate(zip(self.networks_model, self.resume_path)):
            if osp.exists(resume):
                assert osp.exists(resume), 'the resume not exits'
                print('loading...{}'.format(self.networks_name[index]) + self.resume_path[index])
                self.load_network(network, resume, ifprint=False)

        ######################################
        # HELPER COMMENTS
        # optimizer_G/optimizer_D gets set according to train mode
        # for joint training optimizer_G is parsing,appearance,face network and optimizer_D is the appearance network
        ######################################

        self.optimizer_gmm = torch.optim.Adam(
            self.gmm_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        self.optimizer_parsing_G = torch.optim.Adam(
            self.generator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_parsing_D = torch.optim.Adam(
            self.discriminator_parsing.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        self.optimizer_appearance_G = torch.optim.Adam(
            self.generator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_appearance_D = torch.optim.Adam(
            self.discriminator_appearance.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        self.optimizer_face_G = torch.optim.Adam(
            self.generator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])
        self.optimizer_face_D = torch.optim.Adam(
            self.discriminator_face.parameters(), lr=opt.lr, betas=[opt.beta1, 0.999])

        if opt.train_mode == 'gmm':
            self.optimizer_G = self.optimizer_gmm

        elif opt.joint_all:
            self.optimizer_G = [self.optimizer_gmm, self.optimizer_parsing_G,
                                self.optimizer_appearance_G, self.optimizer_face_G]
            setattr(self, 'optimizer_D', getattr(
                self, 'optimizer_' + self.train_mode + '_D'))

        else:
            setattr(self, 'optimizer_G', getattr(
                self, 'optimizer_' + self.train_mode + '_G'))
            setattr(self, 'optimizer_D', getattr(
                self, 'optimizer_' + self.train_mode + '_D'))

        # Tensorboard

        if opt.train_mode == 'gmm':
            if not osp.exists(osp.join(self.save_dir, 'GMM_tboard')):
                os.makedirs(osp.join(self.save_dir, 'GMM_tboard'))
            self.writer = SummaryWriter(osp.join(self.save_dir, 'GMM_tboard'))

        elif opt.train_mode == 'parsing':
            if not osp.exists(osp.join(self.save_dir, 'parsing_tboard')):
                os.makedirs(osp.join(self.save_dir, 'parsing_tboard'))
            self.writer = SummaryWriter(
                osp.join(self.save_dir, 'parsing_tboard'))

        elif opt.train_mode == 'appearance':

            if opt.joint_all:
                if not osp.exists(osp.join(self.save_dir, 'joint_tboard')):
                    os.makedirs(osp.join(self.save_dir, 'joint_tboard'))
                self.writer = SummaryWriter(
                    osp.join(self.save_dir, 'joint_tboard'))
            else:
                if not osp.exists(osp.join(self.save_dir, 'appearance_tboard')):
                    os.makedirs(osp.join(self.save_dir, 'appearance_tboard'))
                self.writer = SummaryWriter(
                    osp.join(self.save_dir, 'appearance_tboard'))

        elif opt.train_mode == 'face':
            if not osp.exists(osp.join(self.save_dir, 'face_tboard')):
                os.makedirs(osp.join(self.save_dir, 'face_tboard'))
            self.writer = SummaryWriter(
                osp.join(self.save_dir, 'face_tboard'))

        self.t1 = time()

    # Set the inputs according to the models
    def set_input(self, opt, result):

        self.t2 = time()

        # Input data returned by dataloader
        self.source_pose_embedding = result['source_pose_embedding'].float(
        ).cuda()
        self.target_pose_embedding = result['target_pose_embedding'].float(
        ).cuda()
        self.source_densepose_data = result['source_densepose_data'].float(
        ).cuda()
        # self.source_densepose_vis = result['source_densepose_vis'].float(
        # ).cuda()
        self.target_densepose_data = result['target_densepose_data'].float(
        ).cuda()
        # self.target_densepose_vis = result['target_densepose_vis'].float(
        # ).cuda()
        self.source_image = result['source_image'].float().cuda()
        self.target_image = result['target_image'].float().cuda()
        self.source_parse = result['source_parse'].float().cuda()
        self.target_parse = result['target_parse'].float().cuda()
        self.cloth_image = result['cloth_image'].float().cuda()
        self.cloth_parse = result['cloth_parse'].float().cuda()
        # self.warped_cloth = result['warped_cloth_image'].float().cuda() # preprocess warped image from gmm model
        self.target_parse_cloth = result['target_parse_cloth'].float().cuda()
        # self.target_pose_img = result['target_pose_img']
        self.image_without_cloth = create_part(
            self.source_image, self.source_parse, 'image_without_cloth', False)
        self.im_c = result['im_c'].float().cuda()  # target warped cloth

        if opt.train_mode != 'parsing':
            # Input for gmm Model
            self.im_h = result['im_h'].float().cuda()
            self.source_parse_shape = result['source_parse_shape'].float(
            ).cuda()
            self.agnostic = torch.cat(
                (self.source_parse_shape, self.im_h, self.source_densepose_data), dim=1) # 31

        # Input for parsing Model
        if opt.train_mode != 'gmm':
            index = [x for x in list(range(20)) if x !=
                     5 and x != 6 and x != 7]
            self.source_parse_tformed = result['source_parse_tformed'].float().cuda()
            real_s_ = torch.index_select(
                self.source_parse_tformed, 1, torch.tensor(index).cuda())
            self.input_parsing = torch.cat(
                (real_s_, self.target_densepose_data, self.cloth_parse), 1).cuda() # 17 + 27 + 1 = 45

        ######################################
        # Part 1 GMM
        ######################################
        # For GMM training we need agnostic cloth_represent(source_head, densepose) original_cloth (from dataloader)
        if opt.train_mode == 'gmm':
            pass

        ######################################
        # Part 2 PARSING
        ######################################
        # For parsing training
        # Input  input_parsing
        # output is the target parse
        if opt.train_mode == 'parsing':
            self.real_s = self.input_parsing
            self.source_parse_vis = result['source_parse_vis'].float(
            ).cuda()
            self.target_parse_vis = result['target_parse_vis'].float(
            ).cuda()

        ######################################
        # Part 3 APPEARANCE
        ######################################
        # For appearance training
        # Input generated parse + warped_cloth + generated_parsing
        # Output corse render image(compare with target image) and composition mask (compare with warped_cloth_parse(this is generated from parsing network))

        if opt.train_mode == 'appearance':

            # If join all training then train flow gradients else don't flow
            if opt.joint_all:
                self.grid, self.theta = self.gmm_model(
                    self.agnostic, self.cloth_image)
                self.warped_cloth = F.grid_sample(self.cloth_image, self.grid)
                self.generated_parsing = F.softmax(
                    self.generator_parsing(self.input_parsing), 1)

            else:
                with torch.no_grad():
                    self.grid, self.theta = self.gmm_model(
                        self.agnostic, self.cloth_image)
                    self.warped_cloth = F.grid_sample(
                        self.cloth_image, self.grid)
                    self.generated_parsing = F.softmax(
                        self.generator_parsing(self.input_parsing), 1)

            # Input to the generated appearance network
            self.input_appearance = torch.cat(
                (self.image_without_cloth, self.warped_cloth, self.generated_parsing), 1).cuda()
            "attention please"
            generated_parsing_ = torch.argmax(
                self.generated_parsing, 1, keepdim=True)

            # input to the generator appearance
            self.generated_parsing_argmax = torch.Tensor()

            # create the warped_cloth_parse from the parsing network
            for _ in range(20):
                self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float(
                ).cuda(), (generated_parsing_ == _).float()], dim=1)
            self.warped_cloth_parse = ((generated_parsing_ == 5) + (
                generated_parsing_ == 6) + (generated_parsing_ == 7)).float().cuda()

            # For visualization
            if opt.save_time:
                self.generated_parsing_vis = torch.Tensor(
                    [0]).expand_as(self.target_image)
            else:
                # decode labels cost much time
                _generated_parsing = torch.argmax(
                    self.generated_parsing, 1, keepdim=True)
                _generated_parsing = _generated_parsing.permute(
                    0, 2, 3, 1).contiguous().int()
                self.generated_parsing_vis = pose_utils.decode_labels(
                    _generated_parsing)  # array

            # For gan training
            self.real_s = self.source_image

        ######################################
        # Part 4 FACE
        ######################################

        if opt.train_mode == 'face':
            if opt.joint_all:
                generated_parsing = F.softmax(
                    self.generator_parsing(self.input_parsing), 1)
                self.generated_parsing_face = F.softmax(
                    self.generator_parsing(self.input_parsing), 1)
            else:
                generated_parsing = F.softmax(
                    self.generator_parsing(self.input_parsing), 1)

                "attention please"
                generated_parsing_ = torch.argmax(
                    generated_parsing, 1, keepdim=True)
                self.generated_parsing_argmax = torch.Tensor()

                for _ in range(20):
                    self.generated_parsing_argmax = torch.cat([self.generated_parsing_argmax.float(
                    ).cuda(), (generated_parsing_ == _).float()], dim=1)

                # self.generated_parsing_face = generated_parsing_c
                self.generated_parsing_face = self.target_parse

            self.input_appearance = torch.cat(
                (self.image_without_cloth, self.warped_cloth, generated_parsing), 1).cuda()

            with torch.no_grad():
                self.generated_inter = self.generator_appearance(
                    self.input_appearance)
                p_rendered, m_composite = torch.split(
                    self.generated_inter, 3, 1)
                p_rendered = F.tanh(p_rendered)
                m_composite = F.sigmoid(m_composite)
                self.generated_image = self.warped_cloth * \
                    m_composite + p_rendered * (1 - m_composite)

            self.source_face = create_part(
                self.source_image, self.source_parse, 'face', False)
            self.target_face_real = create_part(
                self.target_image, self.generated_parsing_face, 'face', False)
            self.target_face_fake = create_part(
                self.generated_image, self.generated_parsing_face, 'face', False)
            self.generated_image_without_face = self.generated_image - self.target_face_fake

            self.input_face = torch.cat(
                (self.source_face, self.target_face_fake), 1).cuda()
            self.real_s = self.source_face

        self.t3 = time()

    # All Forward operations of the networks
    def forward(self, opt):
        self.t4 = time()

        ######################################
        # Part 1 GMM Forward
        ######################################
        if self.train_mode == 'gmm':
            self.grid, self.theta = self.gmm_model(
                self.agnostic, self.cloth_image)
            self.warped_cloth_predict = F.grid_sample(
                self.cloth_image, self.grid)

        ######################################
        # Part 2 PARSING Forward
        ######################################
        if opt.train_mode == 'parsing':
            self.fake_t = F.softmax(
                self.generator_parsing(self.input_parsing), dim=1)
            self.real_t = self.target_parse


        ######################################
        # Part 3 APPEARANCE Forward
        ######################################
        if opt.train_mode == 'appearance':
            generated_inter = self.generator_appearance(self.input_appearance)
            p_rendered, m_composite = torch.split(generated_inter, 3, 1)
            p_rendered = F.tanh(p_rendered)
            self.m_composite = F.sigmoid(m_composite)
            p_tryon = self.warped_cloth * self.m_composite + \
                p_rendered * (1 - self.m_composite)
            self.fake_t = p_tryon
            self.real_t = self.target_image

            if opt.joint_all:

                generate_face = create_part(
                    self.fake_t, self.generated_parsing_argmax, 'face', False)
                generate_image_without_face = self.fake_t - generate_face

                real_s_face = create_part(
                    self.source_image, self.source_parse, 'face', False)

                real_t_face = create_part(
                    self.target_image, self.generated_parsing_argmax, 'face', False)
                input = torch.cat((real_s_face, generate_face), dim=1)

                fake_t_face = self.generator_face(input)
                # residual learning
                """attention
                """
                # fake_t_face = create_part(fake_t_face, self.generated_parsing, 'face', False)
                # fake_t_face = generate_face + fake_t_face
                fake_t_face = create_part(
                    fake_t_face, self.generated_parsing_argmax, 'face', False)
                # fake image
                self.fake_t = generate_image_without_face + fake_t_face

        ######################################
        # Part 4 FACE Forward
        ######################################
        if opt.train_mode == 'face':
            self.fake_t = self.generator_face(self.input_face)

            if opt.face_residual:
                self.fake_t = create_part(
                    self.fake_t, self.generated_parsing_face, 'face', False)
                self.fake_t = self.target_face_fake + self.fake_t

            self.fake_t = create_part(
                self.fake_t, self.generated_parsing_face, 'face', False)
            self.refined_image = self.generated_image_without_face + self.fake_t
            self.real_t = create_part(
                self.target_image, self.generated_parsing_face, 'face', False)

        self.t5 = time()

    # All back propagation and loss operations of the GMM and Generator networks
    def backward_G(self, opt):
        self.t6 = time()

        ######################################
        # Part 1 GMM Loss Function
        ######################################

        if opt.train_mode == 'gmm':
            self.loss = self.criterionL1(self.warped_cloth_predict, self.im_c)
            self.loss.backward()
            self.t7 = time()
            return

        else:
            fake_st = torch.cat((self.real_s, self.fake_t), 1)
            pred_fake = self.discriminator(fake_st)

            ######################################
            # Part 2 PARSING Loss Function
            ######################################
            # gan loss + binary cross entropy loss

            if opt.train_mode == 'parsing':
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                self.loss_G_BCE = self.criterionBCE_re(
                    self.fake_t, self.real_t) * opt.lambda_L1

                self.loss_G = self.loss_G_GAN + self.loss_G_BCE
                self.loss_G.backward()

            ######################################
            # Part 3 APPEARANCE Loss Function
            ######################################
            # loss l1 + loss gan + loss mask + loss vgg
            # loss l1 + loss gan + loss mask + loss vgg  + loss parsing if join_all training mode
            if opt.train_mode == 'appearance':
                self.loss_G_GAN = self.criterionGAN(
                    pred_fake, True) * opt.G_GAN
                # vgg_loss
                loss_vgg1, _ = self.criterion_vgg(
                    self.fake_t, self.real_t, self.target_parse, False, True, False)
                loss_vgg2, _ = self.criterion_vgg(
                    self.fake_t, self.real_t, self.target_parse, False, False, False)
                self.loss_G_vgg = (loss_vgg1 + loss_vgg2) * opt.G_VGG
                self.loss_G_mask = self.criterionL1(
                    self.m_composite, self.warped_cloth_parse) * opt.mask
                if opt.mask_tvloss:
                    self.loss_G_mask_tv = self.criterion_tv(self.m_composite)
                else:
                    self.loss_G_mask_tv = torch.Tensor([0]).cuda()
                self.loss_G_L1 = self.criterion_smooth_L1(
                    self.fake_t, self.real_t) * opt.lambda_L1

                if opt.joint_all and opt.joint_parse_loss:
                    self.loss_G_parsing = self.criterionBCE_re(
                        self.generated_parsing, self.target_parse) * opt.joint_G_parsing
                    self.loss_gmm = self.criterionL1(
                        self.warped_cloth, self.im_c)
                    self.loss_G = self.loss_G_GAN + self.loss_G_L1 + \
                        self.loss_G_vgg + self.loss_G_mask + self.loss_G_parsing + self.loss_gmm

                else:
                    self.loss_G = self.loss_G_GAN + self.loss_G_L1 + \
                        self.loss_G_vgg + self.loss_G_mask + self.loss_G_mask_tv
                self.loss_G.backward()

            ######################################
            # Part 4 FACE Loss Function
            ######################################
            if opt.train_mode == 'face':
                _, self.loss_G_vgg = self.criterion_vgg(
                    self.fake_t, self.real_t, self.generated_parsing_face, False, False, False)  # part, gram, neareast
                self.loss_G_vgg = self.loss_G_vgg * opt.face_vgg
                self.loss_G_L1 = self.criterionL1(
                    self.fake_t, self.real_t) * opt.face_L1
                self.loss_G_GAN = self.criterionGAN(
                    pred_fake, True) * opt.face_gan
                self.loss_G_refine = self.criterionL1(
                    self.refined_image, self.target_image) * opt.face_img_L1

                self.loss_G = self.loss_G_vgg + self.loss_G_L1 + \
                    self.loss_G_GAN + self.loss_G_refine
                self.loss_G.backward()

        self.t7 = time()

   # All back propagation and loss operations Discriminator networks
    def backward_D(self, opt):
        self.t8 = time()

        fake_st = torch.cat((self.real_s, self.fake_t), 1)
        real_st = torch.cat((self.real_s, self.real_t), 1)
        pred_fake = self.discriminator(fake_st.detach())
        pred_real = self.discriminator(real_st)
        
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        self.loss_D.backward()

        self.t9 = time()

    # All optimizer operations of the networks
    def optimize_parameters(self, opt):

        self.t10 = time()

        # Forward Function
        self.forward(opt)

        ######################################
        # Part 1 GMM Optimizer
        ######################################

        if opt.train_mode == 'gmm':
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G(opt)                # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
            self.t11 = time()
            return

        ######################################
        # Part 2 PARSING/APPEARANCE/FACE Network Optimizer For Generator And Discriminator
        ######################################
        else:
            # Update the discriminator
            # enable backprop for D
            self.set_requires_grad(self.discriminator, True)
            self.optimizer_D.zero_grad()                      # set D's gradients to zero
            # calculate gradients for D
            self.backward_D(opt)
            self.optimizer_D.step()                           # update D's weights

            # update the generator
            # D requires no gradients when optimizing G
            self.set_requires_grad(self.discriminator, False)
            if opt.joint_all:
                for _ in self.optimizer_G:
                    _.zero_grad()

                self.backward_G(opt)

                for _ in self.optimizer_G:
                    _.step()
            else:
                self.optimizer_G.zero_grad()        # set G's gradients to zero
                self.backward_G(opt)                # calculate graidents for G
                self.optimizer_G.step()             # udpate G's weights

        self.t11 = time()

    # Saving the images for visualization for training and testing purposes
    def save_result(self, test_data_loader, opt, epoch, iteration):
        for index, test_data in enumerate(test_data_loader):

            # set the data
            with torch.no_grad():  # TODO Should this be torch.no_grad() or this is unnecessary
                img_name = test_data['source_image_name'][0].split('.')[0] # save the image by it's name
                self.set_input(opt, test_data)
                # call forward mode
                self.forward(opt)

            ######################################
            # Part 1 GMM Results
            ######################################
            if opt.train_mode == 'gmm':
                images = [self.source_image, self.cloth_image, self.im_c,
                          self.warped_cloth_predict.detach()]

            ######################################
            # Part 2 PARSING Results
            ######################################
            if opt.train_mode == 'parsing':
                fake_t_vis = pose_utils.decode_labels(torch.argmax(
                    self.fake_t, dim=1, keepdim=True).permute(0, 2, 3, 1).contiguous())
                test_me = pose_utils.decode_labels(torch.argmax(
                    self.source_parse_tformed, dim=1, keepdim=True).permute(0, 2, 3, 1).contiguous())
                images = [test_me, self.target_parse_vis, fake_t_vis]
                # for i in images:
                #     print(i.is_cuda)

            ######################################
            # Part 3 APPEARANCE Results
            ######################################
            if opt.train_mode == 'appearance':
                images = [self.image_without_cloth, self.warped_cloth.detach(), self.warped_cloth_parse, self.target_image,
                          self.cloth_image, self.generated_parsing_vis, self.fake_t.detach()]

            ######################################
            # Part 4 FACE Results
            ######################################
            if opt.train_mode == 'face':
                images = [self.generated_image.detach(), self.refined_image.detach(
                ), self.source_image, self.target_image, self.real_t, self.fake_t.detach()]

            pose_utils.save_img(images, os.path.join(
                self.vis_path, str(img_name) + '_' + str(epoch) + '_' + str(iteration) + '.jpg'))

    # Save the trained models
    def save_model(self, opt, epoch):

        ######################################
        # Part 1 GMM Model Save
        ######################################
        if opt.train_mode == 'gmm':
            model_G = osp.join(self.save_dir, 'generator',
                               'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss))

            if not osp.exists(osp.join(self.save_dir, 'generator')):
                os.makedirs(osp.join(self.save_dir, 'generator'))
            torch.save(self.generator.state_dict(), model_G)

        ######################################
        # Part 2 PARSING/APPEARANCE/FACE Model Save
        ######################################
        elif not opt.joint_all:
            model_G = osp.join(self.save_dir, 'generator',
                               'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_G))
            model_D = osp.join(self.save_dir, 'dicriminator',
                               'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_D))
            if not osp.exists(osp.join(self.save_dir, 'generator')):
                os.makedirs(osp.join(self.save_dir, 'generator'))
            if not osp.exists(osp.join(self.save_dir, 'dicriminator')):
                os.makedirs(osp.join(self.save_dir, 'dicriminator'))

            torch.save(self.generator.state_dict(), model_G)
            torch.save(self.discriminator.state_dict(), model_D)

        ######################################
        # Part 2 JOINT  Model Save
        ######################################
        else:
            model_g_gmm = osp.join(self.save_dir, 'gmm_generator',
                                   'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_G))
            model_G_parsing = osp.join(self.save_dir, 'generator_parsing',
                                       'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_G))
            model_D_parsing = osp.join(self.save_dir, 'dicriminator_parsing',
                                       'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_D))

            model_G_appearance = osp.join(self.save_dir, 'generator_appearance',
                                          'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_G))
            model_D_appearance = osp.join(self.save_dir, 'dicriminator_appearance',
                                          'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_D))

            model_G_face = osp.join(self.save_dir, 'generator_face',
                                    'checkpoint_G_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_G))
            model_D_face = osp.join(self.save_dir, 'dicriminator_face',
                                    'checkpoint_D_epoch_%d_loss_%0.5f_pth.tar' % (epoch, self.loss_D))

            joint_save_dirs = [osp.join(self.save_dir, 'gmm_generator'), osp.join(self.save_dir, 'generator_parsing'), osp.join(self.save_dir, 'dicriminator_parsing'),
                               osp.join(self.save_dir, 'generator_appearance'), osp.join(
                                   self.save_dir, 'dicriminator_appearance'),
                               osp.join(self.save_dir, 'generator_face'), osp.join(self.save_dir, 'dicriminator_face')]
            for _ in joint_save_dirs:
                if not osp.exists(_):
                    os.makedirs(_)
            torch.save(self.generator_parsing.state_dict(), model_G_parsing)
            torch.save(self.generator_appearance.state_dict(),
                       model_G_appearance)
            torch.save(self.generator_face.state_dict(), model_G_face)
            torch.save(self.discriminator_appearance.state_dict(),
                       model_D_appearance)

    # Print the logs while training
    def print_current_errors(self, opt, epoch, i, iteration):

        ######################################
        # Part 1 GMM Print Logs
        ######################################

        if opt.train_mode == 'gmm':

            errors = {'loss_L1': self.loss.item()}

            for key in errors:
                self.writer.add_scalar('Loss/GMM/loss_L1/' + str(key),
                                       errors[key], iteration)
        ######################################
        # Part 2 PARSING Print Logs
        ######################################
        if opt.train_mode == 'parsing':

            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_BCE': self.loss_G_BCE.item(),
                      'loss_D': self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake': self.loss_D_fake.item()}

            for key in errors:
                self.writer.add_scalar('Loss/PARSING/' + str(key),
                                       errors[key], iteration)

        ######################################
        # Part 3 APPEARANCE Print Logs
        ######################################
        if opt.train_mode == 'appearance':

            if opt.joint_all and opt.joint_parse_loss:
                errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg': self.loss_G_vgg.item(), 'loss_G_mask': self.loss_G_mask.item(),
                          'loss_G_L1': self.loss_G_L1.item(), 'loss_D': self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake': self.loss_D_real.item(),
                          'loss_G_parsing': self.loss_G_parsing.item(), 'loss_gmm': self.loss_gmm.item()}
                for key in errors:
                    self.writer.add_scalar(
                        'Loss/JOINTALL/' + str(key), errors[key], iteration)
            else:
                errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg': self.loss_G_vgg.item(), 'loss_G_mask': self.loss_G_mask.item(),
                          'loss_G_L1': self.loss_G_L1.item(), 'loss_D': self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake': self.loss_D_real.item(), 'loss_G_mask_tv': self.loss_G_mask_tv.item()}

                for key in errors:
                    self.writer.add_scalar(
                        'Loss/APPEARANCE/' + str(key), errors[key], iteration)

        ######################################
        # Part 4 FACE Print Logs
        ######################################
        if opt.train_mode == 'face':

            errors = {'loss_G': self.loss_G.item(), 'loss_G_GAN': self.loss_G_GAN.item(), 'loss_G_vgg': self.loss_G_vgg.item(), 'loss_G_refine': self.loss_G_refine.item(),
                      'loss_G_L1': self.loss_G_L1.item(), 'loss_D': self.loss_D.item(), 'loss_D_real': self.loss_D_real.item(), 'loss_D_fake': self.loss_D_real.item()}

            for key in errors:
                self.writer.add_scalar('Loss/FACE/' + str(key),
                                       errors[key], iteration)

        # Print the errors
        t = self.t11 - self.t2
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in sorted(errors.items()):
            if v != 0:
                message += '%s: %.3f ' % (k, v)
        print(message)

        # Save logs
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

from IPython.terminal.embed import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodules import *
from .modules import *



def epipolar_feature(self, feats, proj_mats, patch_idx, depth_samps, depth_samps_2d, cost_reg, is_training=False):


    proj_mats = torch.unbind(proj_mats, 1)
    num_views = len(feats)
    num_depth = depth_samps.shape[1]

    assert len(proj_mats) == num_views

    ref_feat, src_feats = feats[0], feats[1:]
    ref_proj, src_projs = proj_mats[0], proj_mats[1:]

    ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume


    feature_2d = []
    feature_2d.append(ref_feat.unsqueeze(2).repeat(1, 1, num_depth*2, 1, 1))
    

    for src_fea, src_proj in zip(src_feats, src_projs):
        src_proj_new = src_proj[:, 0].clone()
        src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])

        ref_proj_new = ref_proj[:, 0].clone()
        ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samps)
        warped_volume_2d = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_samps_2d)
        feature_2d.append(warped_volume_2d)


        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2) 
        del warped_volume
    
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))
    prob_volume_pre = cost_reg(volume_variance)
    feature_3d = prob_volume_pre



    feature_2d=torch.stack(feature_2d,dim=1)
    h=feature_2d.shape[4]
    w=feature_2d.shape[5]
    if patch_idx==0:
        feature_3d=feature_3d[:,:,:,0:h//2,0:w//2]
        feature_2d=feature_2d[:,:,:,:,0:h//2,0:w//2]
        depth_samps=depth_samps[:,:,0:h//2,0:w//2]
    if patch_idx==1:
        feature_3d=feature_3d[:,:,:,0:h//2,w//2:w]
        feature_2d=feature_2d[:,:,:,:,0:h//2,w//2:w]
        depth_samps=depth_samps[:,:,0:h//2,w//2:w]

    if patch_idx==2:
        feature_3d=feature_3d[:,:,:,h//2:h,0:w//2]
        feature_2d=feature_2d[:,:,:,:,h//2:h,0:w//2]
        depth_samps=depth_samps[:,:,h//2:h,0:w//2]

    if patch_idx==3:
        feature_3d=feature_3d[:,:,:,h//2:h,w//2:w]
        feature_2d=feature_2d[:,:,:,:,h//2:h,w//2:w]
        depth_samps=depth_samps[:,:,h//2:h,w//2:w]

    set=feature_2d.reshape(1,feature_2d.shape[1],8,-1).squeeze(0).permute(2,0,1)
    
    feature_new=self.tr1(set[:,:,:], set[:,:,:])
    feature_new=self.tr2(feature_new[:,:,:], feature_new[:,:,:])
    feature_new=self.tr3(feature_new[:,:,:], feature_new[:,:,:])
    feature_new=self.tr4(feature_new[:,:,:], feature_new[:,:,:])
    feature_2d=feature_new.permute(1,2,0).view(feature_2d.shape[1],8,16,depth_samps_2d.shape[2]//2,depth_samps_2d.shape[3]//2).unsqueeze(0)


    del set,feature_new
    del proj_mats,ref_feat,ref_proj_new,src_fea,src_feats,src_proj,src_proj_new,volume_sq_sum,volume_sum,feats
    torch.cuda.empty_cache()

    feature_2d_avg=torch.mean(feature_2d,dim=1)
    feature_2d_avg_2=torch.mean(feature_2d**2,dim=1)
    feature_2d_ref=feature_2d[:,0,:,:,:,:]
    feature_2d_var=feature_2d_avg_2-(feature_2d_avg**2)

    feature_2d=torch.cat((feature_2d_avg,feature_2d_ref,feature_2d_var), dim=1)

    return {'feature_2d': feature_2d, 'feature_3d': feature_3d, 'depth_samps':depth_samps}



class RayMVSNet(nn.Module):
    def __init__(self, lamb=1.5, stage_configs=[64, 32, 8],  base_chs=[8, 8, 8], feat_ext_ch=8):
        super(RayMVSNet, self).__init__()

        self.stage_configs = stage_configs

        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(stage_configs)
        self.ds_ratio = {"stage1": 4.0,
                         "stage2": 2.0,
                         "stage3": 1.0
                         }

        self.feature_extraction = FeatExtNet(base_channels=feat_ext_ch)

        self.lstm = RelationEncoder_LSTM(26,50, use_cuda_wyl = True)
        
        self.coarse=CoarseMVSNet(ds_ratio=self.ds_ratio, 
                                stage_configs=self.stage_configs,
                                cost_regularization=nn.ModuleList([CostRegNet(in_channels=self.feature_extraction.out_channels[i],
                                                             base_channels=self.base_chs[i]) for i in range(self.num_stage)]),
                                lamb=self.lamb)
        d_model = 8
        self.tr1 = TransformerLayer(d_model, 1)
        self.tr2 = TransformerLayer(d_model, 1)
        self.tr3 = TransformerLayer(d_model, 1)
        self.tr4 = TransformerLayer(d_model, 1)

        self.FC =nn.Sequential(
            SharedMLP(50, (32,16)),
            nn.Conv1d(16, 1, 1, bias=False),
        )



    def forward(self, imgs, proj_matrices, depth_values, depth_gt, patch_idx):
                features = []
                for nview_idx in range(imgs.shape[1]):
                    img = imgs[:, nview_idx]
                    features.append(self.feature_extraction(img))

                outputs = {}
                depth, cur_depth, exp_var = None, None, None
                outputs=self.coarse.forward_depth(depth, cur_depth, exp_var,proj_matrices, depth_values,features,img,outputs)


                features_stage = [feat["stage3"] for feat in features]
                proj_matrices_stage = proj_matrices["stage3"]
                stage_scale = self.ds_ratio["stage3"]
                cur_h = img.shape[2] // int(stage_scale)
                cur_w = img.shape[3] // int(stage_scale)


                cur_depth = outputs['stage2']['depth'].detach()
                exp_var = outputs['stage2']['variance'].detach()
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                    [cur_h, cur_w], mode='bilinear')
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear')

                depth_range_samples, x, depth_range_samples_2d = samples(cur_depth=cur_depth,
                                                                exp_var=exp_var,
                                                                ndepth=self.stage_configs[2])


                outputs_stage = epipolar_feature(self, features_stage, proj_matrices_stage, patch_idx,
                                            depth_samps=depth_range_samples,
                                            depth_samps_2d=depth_range_samples_2d,
                                            cost_reg=self.coarse.forward_epipolar,
                                            is_training=self.training)
                
                feature_2d = outputs_stage['feature_2d']
                feature_3d = outputs_stage['feature_3d']
                feature_3d=feature_3d.unsqueeze(3).repeat(1,1,1,2,1,1).view(1,1,-1,cur_h//2,cur_w//2)
                
                x = x.unsqueeze(1)[:,:,:,0:cur_h//2,0:cur_w//2].cuda()
                
                line_point_feature = torch.cat((feature_2d,feature_3d,x),dim=1)
                del feature_2d, feature_3d, features_stage
                del depth_range_samples_2d,depth_values,exp_var,features,img,imgs,proj_matrices,proj_matrices_stage
                torch.cuda.empty_cache()

                line_point_feature = line_point_feature.view(1,26,self.stage_configs[2]*2,-1).transpose(1,3).contiguous().squeeze(0)

                tmp_lstm, tmp_sdf = self.lstm(line_point_feature)
     
                sdf_feature=tmp_sdf.unsqueeze(0)      

                lstm_feature=tmp_lstm.unsqueeze(0).transpose(2,1)  
                del tmp_lstm, tmp_sdf,line_point_feature
                torch.cuda.empty_cache()

                final_proportion = self.FC(lstm_feature).view(1, 1, cur_h//2, cur_w//2).squeeze(1)

                depth_range_samples=outputs_stage['depth_samps']
                line_point_start=depth_range_samples[:,0,:,:]
                line_point_end=depth_range_samples[:,self.stage_configs[2]-1,:,:]
                final_depth = (line_point_start + (line_point_end - line_point_start) * final_proportion) 


                outputs_stage['sdf_feature'] = sdf_feature
                outputs_stage['final_depth'] = final_depth
                outputs_stage['final_proportion'] = final_proportion

                if patch_idx==0:
                    depth_gt=depth_gt[:,0:cur_h//2,0:cur_w//2]
                if patch_idx==1:
                    depth_gt=depth_gt[:,0:cur_h//2,cur_w//2:cur_w]

                if patch_idx==2:
                    depth_gt=depth_gt[:,cur_h//2:cur_h,0:cur_w//2]

                if patch_idx==3:
                    depth_gt=depth_gt[:,cur_h//2:cur_h,cur_w//2:cur_w]

                propotion_gt = (depth_gt-line_point_start)/(line_point_end-line_point_start+1e-6)

                x_scale=2*x-1   #[-1,1]
                propotion_gt_sdf=2*propotion_gt-1
                x_propotion_gt_distance = x_scale-propotion_gt_sdf   
                
                scale=torch.max(torch.abs(x_propotion_gt_distance),dim=2)
                x_sdf=torch.div(x_propotion_gt_distance, scale[0])
                sdf_gt=x_sdf                

                outputs_stage['sdf_gt'] = sdf_gt
                outputs["stage_ray"] = outputs_stage
                torch.cuda.empty_cache()
        
                return outputs


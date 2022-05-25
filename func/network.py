# model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class ResModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1):
        super(ResModule, self).__init__()
        self.batchnorm_module=nn.BatchNorm3d(num_features=in_channels)
        self.conv_module=nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        h=F.relu(self.batchnorm_module(x))
        h=self.conv_module(h)
        return h+x


class ResModule_w_groupnorm(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, dilation=1):
        super(ResModule_w_groupnorm, self).__init__()
        self.groupnorm_module = nn.GroupNorm(1, in_channels)
        self.conv_module=nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        h=F.relu(self.groupnorm_module(x))
        h=self.conv_module(h)
        return h+x


class CellSegNet_basic_lite(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func = "softmax"):
        super(CellSegNet_basic_lite, self).__init__()
        
        self.conv1=nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2=nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1=nn.BatchNorm3d(num_features=32)
        self.conv3=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1=ResModule(64, 64)
        self.conv4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2=ResModule(64, 64)
        self.conv5=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3=ResModule(64, 64)
        
        self.deconv1=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2=nn.BatchNorm3d(num_features=64)
        self.deconv2=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3=nn.BatchNorm3d(num_features=64)
        self.deconv3=nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4=nn.BatchNorm3d(num_features=32)
        self.conv6=nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.output_func = output_func
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))
        
        h = self.conv3(c1)
        c2 = self.resmodule1(h)
        
        h = self.conv4(c2)
        c3 = self.resmodule2(h)
        
        h = self.conv5(c3)
        c4 = self.resmodule3(h)
        
        c4 = self.deconv1(c4)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape=c3.shape

        delta_c4_x=int(np.floor((c4.shape[2]-c3_shape[2])/2))
        delta_c4_y=int(np.floor((c4.shape[3]-c3_shape[3])/2))
        delta_c4_z=int(np.floor((c4.shape[4]-c3_shape[4])/2))
        c4 = c4[:, :,
                delta_c4_x:c3_shape[2]+delta_c4_x,
                delta_c4_y:c3_shape[3]+delta_c4_y,
                delta_c4_z:c3_shape[4]+delta_c4_z]
        
        h = c4 + c3
        
        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape=c2.shape
        delta_c2_2_x=int(np.floor((c2_2.shape[2]-c2_shape[2])/2))
        delta_c2_2_y=int(np.floor((c2_2.shape[3]-c2_shape[3])/2))
        delta_c2_2_z=int(np.floor((c2_2.shape[4]-c2_shape[4])/2))
        c2_2 = c2_2[:, :,
                delta_c2_2_x:c2_shape[2]+delta_c2_2_x,
                delta_c2_2_y:c2_shape[3]+delta_c2_2_y,
                delta_c2_2_z:c2_shape[4]+delta_c2_2_z]
        
        h = c2_2 + c2
        
        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape=c1.shape
        delta_c1_2_x=int(np.floor((c1_2.shape[2]-c1_shape[2])/2))
        delta_c1_2_y=int(np.floor((c1_2.shape[3]-c1_shape[3])/2))
        delta_c1_2_z=int(np.floor((c1_2.shape[4]-c1_shape[4])/2))
        c1_2 = c1_2[:, :,
                delta_c1_2_x:c1_shape[2]+delta_c1_2_x,
                delta_c1_2_y:c1_shape[3]+delta_c1_2_y,
                delta_c1_2_z:c1_shape[4]+delta_c1_2_z]
        
        h = c1_2 + c1
        
        h = self.conv6(h)
        
        output = F.softmax(h, dim=1)
        
        return output


class CellSegNet_basic_lite_w_groupnorm(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_lite_w_groupnorm, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule_w_groupnorm(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule_w_groupnorm(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule_w_groupnorm(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.GroupNorm(1, 64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.GroupNorm(1, 64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.GroupNorm(1, 32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4 = self.resmodule3(h)

        c4 = self.deconv1(c4)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        h = self.conv6(h)

        output = F.softmax(h, dim=1)

        return output


class EdgeGatedLayer(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=1):
        super(EdgeGatedLayer, self).__init__()
        self.upsample_edge = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_edge = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv_main = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
    def forward(self, x_edge, x_main):
        x_edge = self.upsample_edge(x_edge)

        x_e = self.conv_edge(x_edge)
        x_m = self.conv_main(x_main)

        alpha = x_e + x_m
        alpha = F.relu(alpha)
        alpha = F.sigmoid(alpha)
        x_out = x_edge * alpha + x_edge
        return x_out


class EdgeGatedLayer_II(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=1):
        super(EdgeGatedLayer_II, self).__init__()
        self.conv_edge = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.conv_main = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x_edge, x_main):
        x_e = self.conv_edge(x_edge)
        x_m = self.conv_main(x_main)

        alpha = x_e + x_m
        alpha = F.relu(alpha)
        alpha = F.sigmoid(alpha)
        x_out = x_edge * alpha + x_edge
        return x_out


class CellSegNet_basic_edge_gated_I(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_I, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        # edge gated here
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        # edge gated here
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)
        # edge conv1 here
        # edge gated here
        # edge conv2 here

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        # TODO: group norm?
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.edgegatelayer1 = EdgeGatedLayer(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer(64, 64)
        self.edge_conv1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.edgegatelayer3 = EdgeGatedLayer(32, 32)
        self.edge_conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.e_output = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1 ,padding=1)

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4 = self.resmodule3(h)

        # edge gated network
        e1 = self.edgegatelayer1(c4, c3)

        e2 = self.edgegatelayer2(e1, c2)

        e_conv_1 = self.edge_conv1(e2)
        e3 = self.edgegatelayer3(e_conv_1, c1)
        conv_e3 = self.edge_conv2(e3)

        c4 = self.deconv1(c4)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape
        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        # finally add edge gated stream to decoder
        h = c1_2 + c1 + e3

        h = self.conv6(h)

        output = F.softmax(h, dim=1)

        e_out = self.e_output(conv_e3)
        e_output = F.softmax(e_out, dim=1)

        return output, e_output


class CellSegNet_basic_edge_gated(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        # edge gated here
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        # edge gated here
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)
        # edge conv1 here
        # edge gated here
        # edge conv2 here

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        # TODO: group norm?
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.resmodule_edge_1 = ResModule(64, 64)
        self.edgegatelayer1 = EdgeGatedLayer(64, 64)
        self.resmodule_edge_2 = ResModule(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer(64, 64)
        self.edge_conv1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=1, stride=1)
        self.resmodule_edge_3 = ResModule(32, 32)
        self.edgegatelayer3 = EdgeGatedLayer(32, 32)
        # self.edge_conv2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1, stride=1)

        self.e_output = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4 = self.resmodule3(h)

        # edge gated network
        res_edge_1 = self.resmodule_edge_1(c4)
        e1 = self.edgegatelayer1(res_edge_1, c3)

        res_edge_2 = self.resmodule_edge_2(e1)
        e2 = self.edgegatelayer2(res_edge_2, c2)

        e_conv_1 = self.edge_conv1(e2)
        res_edge_3 = self.resmodule_edge_3(e_conv_1)
        e3 = self.edgegatelayer3(res_edge_3, c1)

        c4 = self.deconv1(c4)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape
        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        # finally add edge gated stream to decoder
        h = c1_2 + c1

        h = torch.cat((h, e3), dim=1)

        h = self.conv6(h)

        output = F.softmax(h, dim=1)

        e_out = self.e_output(e3)
        e_output = F.softmax(e_out, dim=1)

        return output, e_output


class CellSegNet_basic_edge_gated_II(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_II, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)

        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge_bridge= self.edgegatelayer3(c1_2_edge, c1)

        h_edge = self.conv6_edge(h_edge_bridge)

        output_edge = self.sigmoid_edge(h_edge)


        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)
        h = h + h_edge_bridge
        h = self.conv6(h)

        output = F.softmax(h, dim=1)
        return output, output_edge


class CellSegNet_basic_edge_gated_III(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_III, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)

        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, stride=1, padding=1)

        self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge_bridge= self.edgegatelayer3(c1_2_edge, c1)

        output_edge = self.conv6_edge(h_edge_bridge)

        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)
        h = h + h_edge_bridge
        h = self.conv6(h)

        output = F.softmax(h, dim=1)
        return output, output_edge


class CellSegNet_basic_edge_gated_IV(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_IV, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)

        self.edgegatelayer4 = EdgeGatedLayer_II(32, 32)

        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7_edge = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge = self.edgegatelayer3(c1_2_edge, c1)

        h_edge_bridge = self.conv6_edge(h_edge)
        output_edge = self.conv7_edge(h_edge_bridge)
        output_edge = self.sigmoid_edge(output_edge)

        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)

        h = self.conv6(h)
        h = self.edgegatelayer4(h, h_edge_bridge)

        h = self.conv7(h)

        output = F.softmax(h, dim=1)
        return output, output_edge


class CellSegNet_basic_edge_gated_V(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_V, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm3d(num_features=32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.BatchNorm3d(num_features=64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.BatchNorm3d(num_features=64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.BatchNorm3d(num_features=32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)

        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7_edge = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge = self.edgegatelayer3(c1_2_edge, c1)

        h_edge_bridge = self.conv6_edge(h_edge)
        output_edge = self.conv7_edge(h_edge_bridge)
        output_edge = self.sigmoid_edge(output_edge)

        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)
        h = h + h_edge_bridge
        h = self.conv6(h)

        #h = self.edgegatelayer4(h, h_edge_bridge)

        h = self.conv7(h)

        output = F.softmax(h, dim=1)
        return output, output_edge


class CellSegNet_basic_edge_gated_VI(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_VI, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule_w_groupnorm(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule_w_groupnorm(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule_w_groupnorm(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.GroupNorm(1, 64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.GroupNorm(1, 64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.GroupNorm(1, 32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)

        self.edgegatelayer4 = EdgeGatedLayer_II(32, 32)

        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7_edge = nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=1)

        self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge = self.edgegatelayer3(c1_2_edge, c1)

        h_edge_bridge = self.conv6_edge(h_edge)
        output_edge = self.conv7_edge(h_edge_bridge)
        output_edge = self.sigmoid_edge(output_edge)

        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)

        h = self.conv6(h)
        h = self.edgegatelayer4(h, h_edge_bridge)

        h = self.conv7(h)

        output = F.softmax(h, dim=1)
        return output, output_edge


class CellSegNet_basic_edge_gated_VII(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func="softmax"):
        super(CellSegNet_basic_edge_gated_VII, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=input_channel, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule1 = ResModule_w_groupnorm(64, 64)
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule2 = ResModule_w_groupnorm(64, 64)
        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.resmodule3 = ResModule_w_groupnorm(64, 64)

        self.deconv1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2 = nn.GroupNorm(1, 64)
        self.deconv2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3 = nn.GroupNorm(1, 64)
        self.deconv3 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4 = nn.GroupNorm(1, 32)
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(in_channels=32, out_channels=2, kernel_size=1)

        self.edgegatelayer1 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer2 = EdgeGatedLayer_II(64, 64)
        self.edgegatelayer3 = EdgeGatedLayer_II(32, 32)


        self.deconv1_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm2_edge = nn.BatchNorm3d(num_features=64)
        self.deconv2_edge = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bnorm3_edge = nn.BatchNorm3d(num_features=64)
        self.deconv3_edge = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bnorm4_edge = nn.BatchNorm3d(num_features=32)
        self.conv6_edge = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv7_edge = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)

        # self.sigmoid_edge = nn.Sigmoid()

        self.output_func = output_func

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        c1 = F.relu(self.bnorm1(h))

        h = self.conv3(c1)
        c2 = self.resmodule1(h)

        h = self.conv4(c2)
        c3 = self.resmodule2(h)

        h = self.conv5(c3)
        c4_encoder_end = self.resmodule3(h)

        # decoder
        c4 = self.deconv1(c4_encoder_end)
        c4 = F.relu(self.bnorm2(c4))
        c3_shape = c3.shape

        delta_c4_x = int(np.floor((c4.shape[2] - c3_shape[2]) / 2))
        delta_c4_y = int(np.floor((c4.shape[3] - c3_shape[3]) / 2))
        delta_c4_z = int(np.floor((c4.shape[4] - c3_shape[4]) / 2))
        c4 = c4[:, :,
             delta_c4_x:c3_shape[2] + delta_c4_x,
             delta_c4_y:c3_shape[3] + delta_c4_y,
             delta_c4_z:c3_shape[4] + delta_c4_z]

        h = c4 + c3

        h = self.deconv2(h)
        c2_2 = F.relu(self.bnorm3(h))
        c2_shape = c2.shape
        delta_c2_2_x = int(np.floor((c2_2.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_y = int(np.floor((c2_2.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_z = int(np.floor((c2_2.shape[4] - c2_shape[4]) / 2))
        c2_2 = c2_2[:, :,
               delta_c2_2_x:c2_shape[2] + delta_c2_2_x,
               delta_c2_2_y:c2_shape[3] + delta_c2_2_y,
               delta_c2_2_z:c2_shape[4] + delta_c2_2_z]

        h = c2_2 + c2

        h = self.deconv3(h)
        c1_2 = F.relu(self.bnorm4(h))
        c1_shape = c1.shape
        delta_c1_2_x = int(np.floor((c1_2.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_y = int(np.floor((c1_2.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_z = int(np.floor((c1_2.shape[4] - c1_shape[4]) / 2))
        c1_2 = c1_2[:, :,
               delta_c1_2_x:c1_shape[2] + delta_c1_2_x,
               delta_c1_2_y:c1_shape[3] + delta_c1_2_y,
               delta_c1_2_z:c1_shape[4] + delta_c1_2_z]

        h = c1_2 + c1

        # edge stream
        c4_edge = self.deconv1_edge(c4_encoder_end)
        c4_edge = F.relu(self.bnorm2_edge(c4_edge))
        c3_shape = c3.shape

        delta_c4_edge_x = int(np.floor((c4_edge.shape[2] - c3_shape[2]) / 2))
        delta_c4_edge_y = int(np.floor((c4_edge.shape[3] - c3_shape[3]) / 2))
        delta_c4_edge_z = int(np.floor((c4_edge.shape[4] - c3_shape[4]) / 2))
        c4_edge = c4_edge[:, :,
             delta_c4_edge_x:c3_shape[2] + delta_c4_edge_x,
             delta_c4_edge_y:c3_shape[3] + delta_c4_edge_y,
             delta_c4_edge_z:c3_shape[4] + delta_c4_edge_z]

        h_edge = self.edgegatelayer1(c4_edge, c3)

        h_edge = self.deconv2_edge(h_edge)
        c2_2_edge = F.relu(self.bnorm3_edge(h_edge))
        c2_shape = c2.shape
        delta_c2_2_edge_x = int(np.floor((c2_2_edge.shape[2] - c2_shape[2]) / 2))
        delta_c2_2_edge_y = int(np.floor((c2_2_edge.shape[3] - c2_shape[3]) / 2))
        delta_c2_2_edge_z = int(np.floor((c2_2_edge.shape[4] - c2_shape[4]) / 2))
        c2_2_edge = c2_2_edge[:, :,
               delta_c2_2_edge_x:c2_shape[2] + delta_c2_2_edge_x,
               delta_c2_2_edge_y:c2_shape[3] + delta_c2_2_edge_y,
               delta_c2_2_edge_z:c2_shape[4] + delta_c2_2_edge_z]

        h_edge = self.edgegatelayer2(c2_2_edge, c2)

        h_edge = self.deconv3_edge(h_edge)
        c1_2_edge = F.relu(self.bnorm4_edge(h_edge))
        c1_shape = c1.shape
        delta_c1_2_edge_x = int(np.floor((c1_2_edge.shape[2] - c1_shape[2]) / 2))
        delta_c1_2_edge_y = int(np.floor((c1_2_edge.shape[3] - c1_shape[3]) / 2))
        delta_c1_2_edge_z = int(np.floor((c1_2_edge.shape[4] - c1_shape[4]) / 2))
        c1_2_edge = c1_2_edge[:, :,
               delta_c1_2_edge_x:c1_shape[2] + delta_c1_2_edge_x,
               delta_c1_2_edge_y:c1_shape[3] + delta_c1_2_edge_y,
               delta_c1_2_edge_z:c1_shape[4] + delta_c1_2_edge_z]

        h_edge = self.edgegatelayer3(c1_2_edge, c1)

        h_edge_bridge = self.conv6_edge(h_edge)
        output_edge = self.conv7_edge(h_edge_bridge)
        # output_edge = self.sigmoid_edge(output_edge)

        # main stream
        # h = torch.cat((h, h_edge_bridge), dim=1)

        h = self.conv6(h)
        # h = self.edgegatelayer4(h, h_edge_bridge)

        h = self.conv7(h)

        h = torch.cat((h, output_edge), dim=1)


        output = F.softmax(h, dim=1)
        return output, output_edge

    
class VoxResNet(nn.Module):
    def __init__(self, input_channel=1, n_classes=3, output_func = "softmax"):
        super(VoxResNet, self).__init__()
        
        self.conv1a=nn.Conv3d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1a=nn.BatchNorm3d(num_features=32)
        self.conv1b=nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bnorm1b=nn.BatchNorm3d(num_features=32)
        self.conv1c=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res2=ResModule(64, 64)
        self.res3=ResModule(64, 64)
        self.bnorm3=nn.BatchNorm3d(num_features=64)
        self.conv4=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res5=ResModule(64, 64)
        self.res6=ResModule(64, 64)
        self.bnorm6=nn.BatchNorm3d(num_features=64)
        self.conv7=nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.res8=ResModule(64, 64)
        self.res9=ResModule(64, 64)
        
        self.c1deconv=nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.c1conv=nn.Conv3d(in_channels=32, out_channels=n_classes, kernel_size=3, padding=1)
        self.c2deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.c2conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c3deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=6, stride=4, padding=1)
        self.c3conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        self.c4deconv=nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=10, stride=8, padding=1)
        self.c4conv=nn.Conv3d(in_channels=64, out_channels=n_classes, kernel_size=3, padding=1)
        
        self.output_func = output_func
    def forward(self, x):
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = F.relu6(self.c1deconv(h))
        c1 = self.c1conv(c1)
        
        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.res2(h)
        h = self.res3(h)
        c2 = F.relu6(self.c2deconv(h))
        c2 = self.c2conv(c2)
        
        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.res5(h)
        h = self.res6(h)
        c3 = F.relu6(self.c3deconv(h))
        c3 = self.c3conv(c3)
        
        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.res8(h)
        h = self.res9(h)
        c4 = F.relu6(self.c4deconv(h))
        c4 = self.c4conv(c4)
        
        c = c1 + c2 + c3 + c4
        
        output = F.softmax(c, dim=1)
        
        return output
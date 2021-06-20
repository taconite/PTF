from im2mesh.encoder import conv, pointnet, ifnet

encoder_dict = {
    'resnet18': conv.Resnet18,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'pointnet_conv': pointnet.ConvPointnet,
    'ifnet': ifnet.IFNet,
}

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from collections import OrderedDict
# from utils.layers import AmpNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, **kwargs):
        super(_DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growth_rate, affine=False, track_running_stats=False)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
    
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, **kwargs):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, **kwargs):
        super(_Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, input_shape=[3, 96, 96], growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2,
                 do_norm=False, d_channels=8, **kwargs):

        super(DenseNet, self).__init__()

        # self.amp_norm = AmpNorm(input_shape=input_shape)
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(num_init_features, affine=False, track_running_stats=False)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i == 0:
                self.features.add_module('zero_padding', nn.ZeroPad2d(2))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('bn5', nn.BatchNorm2d(num_features, affine=False, track_running_stats=False))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.do_norm = do_norm
        if self.do_norm:
            self.stain_norm = BeerLaNet(r=d_channels, c=3, learn_S_init=True)
            self.adapt_conv = nn.Conv2d(d_channels, 3, kernel_size=1)

    def forward(self, x):
        # x = self.amp_norm(x)
        if self.do_norm:
            _, _, D = self.stain_norm(x, S=None, D=None, n_iter=10, unit_norm_S=True)
            x = self.adapt_conv(D)

        features = self.features(x)
        out = F.relu(features,inplace=True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        emb = torch.flatten(out, 1)
        out = self.classifier(emb)

        if self.do_norm: return out, x, emb
        return out, -1, emb



class UNet(nn.Module):
    def __init__(self, input_shape, in_channels=3, out_channels=2, init_features=32):
        super(UNet, self).__init__()

        # self.amp_norm = AmpNorm(input_shape=input_shape)

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1",)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        # x = self.amp_norm(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = self.conv(dec1)

        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "_conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn1", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu1", nn.ReLU(inplace=True)),
                    (
                        name + "_conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "_bn2", nn.BatchNorm2d(num_features=features, affine=False, track_running_stats=False)),
                    (name + "_relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BeerLaNet(nn.Module):
    
    def __init__(self, r, c=3, learn_S_init = False, calc_tau = True):
        """
        A layer to perform color normalization based on a sparse, low-rank non-negative
        matrix factorization model of the form:
            
        min_{x_0,S,D} 1/2|X + S*D^T - x_0*1^T|_F^2 + \lambda \sum_{i=1}^r |s_i|_2 (\gamma |d_i|_1 + |d_i|_2)
        s.t. S>=0, D>=0
        
        This is accomplished through alternating updates through each block of
        variables (x_0,S,D).
        
        In the above equation X is assumed to be reshaped to have dimension
        c x p where c is the number of color channels in the image (typically 3
        for RGB images) and p is the number of pixels in the image.  For the
        actual forward operator of the layer, however, X will be input in typical 
        pytorch format of [n,p1,p2,c] where n is the number of images in the mini-batch
        (p1,p2) are the image dimenions, and c is the number of color channels.
        
        In general, the sequence of updates proceeds in order x_0 -> D -> S.
        The initialization for x_0 and D is always taken to be zeros if it is 
        not passed as an input to forward(), while the initialization for S is
        a learned parameter if not passed as an input to forward.
        
        Parameters
        ----------
        r : int
            The number of columns in S and D to use for the layer
            
        c : int
            The first dimension of S (the color spectrum matrix).  
            
        learn_S_init : boolean
            If true then the initialization for S will be a learnable parameter.
            If false, then an initialization for S must be passed as an input
            to the forward function.
            
        calc_tau : boolean
            If true, then the step size parameter (tau) will be calculated based
            on a norm of S/D.  If false, then this parameter will be trainable.
            
        """
        
        super(BeerLaNet, self).__init__()
        self.r = r
        self.c = c
        
        self.calc_tau = calc_tau
        
        self.gamma = nn.Parameter(torch.rand(1)*1e-5)
        self.lam   = nn.Parameter(torch.rand(1)*1e-5)
        
        if not calc_tau:
            self.tau   = nn.Parameter(torch.rand(1)*1e-5)
        
        if learn_S_init:
            #initialize S with uniform variables to be non-negative
            self.S_init = nn.Parameter(torch.rand(self.c,self.r))
            
            with torch.no_grad():
                self.S_init.data = self.S_init.data/self._S_norm(self.S_init.data)
                
        else:
            self.S_init = None        
        
    def forward(self, X, S=None, D=None, n_iter=1, unit_norm_S=True):
        """
        This performs update iterations for the matrix factorization color
        normalization model described in the constructor.  It does this by a 
        sequence of updates to the object in the order x_0 -> D -> S, where
        the update for x_0 is a closed form optimal update, and the updates for
        S and D are via proximal gradient descent updates.

        Parameters
        ----------
        X : pytorch tensor [n,c,p1,p2]
            The input data tensor, where n is the number of minibatch samples,
            (p1,p2) are the image dimensions, and c is the number of color
            channels (or feature channels more generally).  c should be equal
            to the value of c passed to the constructor
            
        S : pytorch tensor [c,r]
            The initial input spectra for the layer.  If this is not provided
            then the layer must have been constructed with learn_S_init=True
        
        D : pytorch tensor [n,r,p1,p2], optional
            The initial input for the density maps.  If this is not provided
            then this will be initialized as all zeros.
        
        n_iter : int, optional
            The number of iterations of the optimization to run in this layer.
            The default is 1.
        
        unit_norm_S : boolean, optional
            If true, then S and D will be rescaled after each iteration so that 
            S has unit norm columns.  Note this does not effect the objective value.

        Returns
        -------
        x_0 : pytorch tensor [c]
            The estimate of the background intensity.
            
        S : pytorch tensor [c,r]
            The current spectrum matrix
            
        D : pytorch tensor [n,r,p1,p2]
            The current optical density maps.

        """

        n,c_in,p1,p2 = X.shape
        p = p1*p2
        
        assert c_in == self.c
        
        #Resahpe the X data to be in matrix form with size [n,c,p]
        X = X.view(-1,self.c,p1*p2)
        
        if S is None:
            S = self.S_init.clone().to(X.device)
        
        if D is None:
            Dt = torch.zeros(n,self.r,p, device=X.device) #This is D^T 
        else:
            Dt = D.view(-1,self.r,p) #This is D^T 
            
        #Make sure the regularization and step size parameters are non-negative
        #Since these are learnable parameters the optimizer can push these
        #negative, so we take the absolute value to prevent this.
        with torch.no_grad():
            self.gamma.data = torch.abs(self.gamma.data)
            self.lam.data   = torch.abs(self.lam.data)
            
            if not self.calc_tau:
                self.tau.data   = torch.abs(self.tau.data)
        
        #Now run the main computation
        
        for _ in range(n_iter):
            SDt = S@Dt # compute S*D^T [n,c,p]
            
            ######################
            #Compute x_0 [n,c,1]
            #We keep the final dimension for broadcasting later
            x_0 = torch.mean(X+SDt,dim=2,keepdims=True)
            
            ######################
            #Now start the updates for D.
            #Here we'll be make the updates with D shaped as D^T
            
            #First the gradient step for Dt
            #Dt = Dt - tau*(S^T*S*D^T + S^T*X - S^T*x_0*1^T)
            
            if self.calc_tau:
                tau_D = 1.0/torch.linalg.matrix_norm(S, ord='fro')**2
            else:
                tau_D = self.tau
            
            Dt += -tau_D*(S.T@SDt + S.T@X - S.T@x_0)
            
            # #Now compute the proximal operator.  This is the composition
            # #of first doing soft-thresholding, followed by scaling
            
            # #First compute the soft-thresholding for the L1 proximal operator
            S_nrm = self._S_norm(S).view(1,self.r,1)
            Dt = F.relu(Dt-self.lam*self.gamma*tau_D*S_nrm)
            
            # #Now compute the scaling for the L2 proximal operator
            Dt_L2 = self._Dt_Lp(Dt,2)
            scl = F.relu(Dt_L2-self.lam*tau_D*S_nrm)
            scl = scl/Dt_L2+1e-10
            Dt = Dt*scl
        
            # ######################
            # #Now the updates for S.
            
            # #First update SDt
            SDt = S@Dt
            
            # #Also update x_0
            x_0 = torch.mean(X+SDt,dim=2,keepdims=True)
            
            # #Now the gradient step for S
            Dt_sum = Dt.sum(dim=2,keepdim=True)
            
            #The gradient step for a single image is given as
            #S = S - tau*(S*D^T*D + X*D - x_0*1^T*D)
            #
            #but here we can have multiple images in a batch, so we take the
            #mean over the batch dimension
            
            if self.calc_tau:
                tau_S = 1.0/torch.mean(torch.linalg.matrix_norm(Dt, ord='fro')**2)
            else:
                tau_S = self.tau
            
            #We rename the variable to avoid inline modification errors for backprop
            S_grad = S-tau_S*torch.mean(SDt@Dt.permute(0,2,1) + X@Dt.permute(0,2,1) - x_0@Dt_sum.permute(0,2,1),
                             dim=0, keepdims=False)
            
            #Now compute the proximal operator for the L2 norm
            #Here we compute the mean norms for Dt across the batch.
            Dt_nrm = self._Dt_norm(Dt).mean(dim=0,keepdims=False) #[r,1]
            S_nrm = self._S_norm(S_grad) #[1,r]
            scl_S = F.relu(S_nrm-self.lam*tau_S*Dt_nrm.T) #[1,r]
            scl_S = scl_S/(S_nrm+1e-10)
            
            S = S_grad*scl_S
            
            # ######################
            # #All the updates are done.
            # #We can rescale to have S with unit norm if desired.
            
            if unit_norm_S:
                 S_nrm = self._S_norm(S)
                 S = S/(S_nrm+1e-10)
                 Dt = Dt*(S_nrm.view(1,self.r,1)+1e-10)
                
                
        return x_0, S, Dt.view(n,self.r,p1,p2)
            
            
    def _S_norm(self,S):
        """
        Returns the L2 norms of S

        Parameters
        ----------
        S : pytorch tensor [c,r]

        Returns
        -------
        norms_S : pytorch tensor [1,r]
        
            norms_S[0,i] corresponds to the L2 norm of the i'th row of S

        """
        
        return torch.linalg.vector_norm(S,ord=2,dim=0,keepdim=True)
    
    def _Dt_Lp(self,Dt,nrm_ord):
        """
        Returns the Lp norm for the columns of D (or the rows of D^T)

        Parameters
        ----------
        Dt : pytorch tensor [n,r,p]
            The tensor containing D^T

        nrm_ord : scaler
            The ord parameter of the norm (see torch.linalg.vector_norm)

        Returns
        -------
        norm_Dt : pytorch tensor [n,r,1]
            norm_L2[i,j,0] corresponds to the Lp norm for the j^th row of the
            optical density map for the i^th image.

        """
        
        return torch.linalg.vector_norm(Dt,ord=nrm_ord,dim=2,keepdim=True)
    
    def _Dt_norm(self,Dt):
        """
        Returns the norm self.gamma||D_i||_1 + ||D_i||_2 for the columns of
        D (or the rows of D^T)

        Parameters
        ----------
        Dt : pytorch tensor [n,r,p]
            The tensor containing D^T

        Returns
        -------
        norms_Dt : pytorch tensor [n,r,1]
            norms_Dt[i,j,0] corresponds to the norm for the j^th row of the
            optical density map for the i^th image.

        """
        
        return self.gamma*self._Dt_Lp(Dt,1) + self._Dt_Lp(Dt,2)
    

class ClientModel(nn.Module):
    def __init__(self, backbone='simple', d_channels=3, num_classes=2, do_norm=False):
        super().__init__()

        self.do_norm = do_norm
        if self.do_norm:
            self.stain_norm = BeerLaNet(r=d_channels, c=3, learn_S_init=True)
            self.adapt_conv = nn.Conv2d(d_channels, 3, kernel_size=1)

        if backbone == 'simple':
            self.backbone = SimpleNetwork()
        elif backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'densenet':
            self.backbone = DenseNet(input_shape=[3, 96, 96], num_classes=2)
        else:
            raise ValueError(f'Expected param `backbone` to be one of [simple, resnet18], got {backbone}')
        
    def forward(self, x):
        if self.do_norm:
            _, _, D = self.stain_norm(x, S=None, D=None, n_iter=10, unit_norm_S=True)
            x = self.adapt_conv(D)
        return self.backbone(x)



if __name__=='__main__':
    exit()
from numpy.lib.function_base import angle
import torch 
import numpy as np
from torch.autograd import Variable

class Perspective_Taker():

    """
    Performs Perspective Taking 
    """

    def __init__(self, num_features, num_dimensions, 
                rotation_gradient_init, translation_gradient_init):

        self.rotation_gradient_init = rotation_gradient_init
        self.translation_gradient_init = translation_gradient_init
        
        self.dimensions = num_dimensions
        self.num_features = num_features

        self.bin_momentum_rotation = [None, None]
        self.bin_momentum_translation = [None, None]


    #################################################################################
    #################### EULER ROTATION
    #################################################################################

    def init_angles_(self):
        return torch.rand((self.dimensions, 1), requires_grad=False) * 360

    
    def compute_rotation_matrix_(self, alpha, beta, gamma):
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        # initialize dimensional rotation matrices 
        R_x_1 = torch.Tensor([[1.0,0.0,0.0]])
        R_x_2 = torch.stack([torch.zeros(1), torch.cos(alpha_rad), - torch.sin(alpha_rad)], dim=1)
        R_x_3 = torch.stack([torch.zeros(1), torch.sin(alpha_rad), torch.cos(alpha_rad)], dim=1)
        R_x = torch.stack([R_x_1, R_x_2, R_x_3], dim=1)
        
        R_y_1 = torch.stack([torch.cos(beta_rad), torch.zeros(1), torch.sin(beta_rad)], dim=1)
        R_y_2 = torch.Tensor([[0.0,1.0,0.0]])
        R_y_3 = torch.stack([- torch.sin(beta_rad), torch.zeros(1), torch.cos(beta_rad)], dim=1)
        R_y = torch.stack([R_y_1, R_y_2, R_y_3], dim=1)

        R_z_1 = torch.stack([torch.cos(gamma_rad), - torch.sin(gamma_rad), torch.zeros(1)], dim=1)
        R_z_2 = torch.stack([torch.sin(gamma_rad), torch.cos(gamma_rad), torch.zeros(1)], dim=1)
        R_z_3 = torch.Tensor([[0.0,0.0,1.0]])
        R_z = torch.stack([R_z_1, R_z_2, R_z_3], dim=1)

        # initialize rotation matrix
        rotation_matrix = torch.matmul(R_z, torch.matmul(R_y, R_x))    

        return rotation_matrix


    def update_rotation_angles_(self, rotation_angles, gradient, learning_rate):
        # update with gradient descent
        upd_rotation_angles = []
        for i in range(self.dimensions):
            with torch.no_grad():
                e = rotation_angles[i] - learning_rate * gradient[i]
                e = e % 360
                upd_rotation_angles.append(e)
        
        return upd_rotation_angles


    def rotate(self, input, rotation_matrix):
        # return torch.matmul(rotation_matrix, input.T).T
        return torch.matmul(rotation_matrix, input.T).T

            
    def update_momentum_rotation(self, entries): 
        if self.bin_momentum_rotation[0] == None: 
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
            self.bin_momentum_rotation[1] = torch.Tensor(entries.clone().detach())
        else: 
            self.bin_momentum_rotation[1] = self.bin_momentum_rotation[0]
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
    

    def calc_momentum_roation(self):
        if self.bin_momentum_rotation[0] == None: 
            return torch.zeros(self.num_features, self.num_features)
        else:
            return self.bin_momentum_rotation[1] - self.bin_momentum_rotation[0]


    def inverse_rotation_angles(self, angles): 
        print(angles)
        reverse_angles = []
        for angle in angles:
            reverse_angles.append(360-angle)
        
        return torch.stack(reverse_angles)
    



    #################################################################################
    #################### QUATERNION ROTATION
    #################################################################################

    def init_quaternion(self): 
        # q = torch.rand(1,4)
        q = torch.ones(1,4)
        q[0,0] = 0.8
        q[0,1] = 0.5
        q[0,2] = 1.2
        q[0,3] = 1.5
        q = self.norm_quaternion(q)

        return q


    def norm_quaternion(self, q): 
        abs = torch.sqrt(torch.sum(torch.mul(q,q)))
        return torch.div(q, abs)


    def update_quaternion(self, q, grad, learning_rate):
        upd_q = q - learning_rate * grad
        upd_q = self.norm_quaternion(upd_q)
        return upd_q

    
    def update_quaternion(self, q, grad, learning_rate, momentum):
        mom = momentum*self.calc_momentum_qroation()
        upd_q = q - learning_rate * grad + mom
        # print(grad)
        # print(mom)
        # print(learning_rate)
        # print(q)
        # print(upd_q)
        upd_q = self.norm_quaternion(upd_q)
        self.update_momentum_qrotation(q)
        return upd_q


    def update_momentum_qrotation(self, entries): 
        if self.bin_momentum_rotation[0] == None: 
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
            self.bin_momentum_rotation[1] = torch.Tensor(entries.clone().detach())
        else: 
            self.bin_momentum_rotation[1] = self.bin_momentum_rotation[0]
            self.bin_momentum_rotation[0] = torch.Tensor(entries.clone().detach())
    

    def calc_momentum_qroation(self):
        if self.bin_momentum_rotation[0] == None: 
            return torch.zeros(1, self.dimensions+1)
        else:
            return self.bin_momentum_rotation[1] - self.bin_momentum_rotation[0]


    def qeuler(self, q, order, epsilon=0):
        """
        Convert quaternion(s) q to Euler angles.
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        assert q.shape[-1] == 4
        
        original_shape = list(q.shape)
        original_shape[-1] = 3
        q = q.view(-1, 4)
        
        q0 = q[:, 0]
        q1 = q[:, 1]
        q2 = q[:, 2]
        q3 = q[:, 3]
        
        if order == 'xyz':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        elif order == 'yzx':
            x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
        elif order == 'zxy':
            x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'xzy':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
            y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
            z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
        elif order == 'yxz':
            x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
            y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
            z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        elif order == 'zyx':
            x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
            y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
            z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
        else:
            raise

        return torch.stack((x, y, z), dim=1).view(original_shape)


    def quaternion2rotmat(self, q):
        if q.size()[0] == 1: 
            q = q[0]

        # get single quaternion entries 
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        # multiplied entries
        xw = x*w
        xx = x*x
        xy = x*y
        xz = x*z

        yw = y*w
        yy = y*y
        yz = y*z

        zw = z*w
        zz = z*z

        # compute roatation matrix
        rotation_matrix = torch.Tensor(
            [[1-2*(yy+zz),   2*(xy-zw),   2*(xz+yw)], 
             [  2*(xy+zw), 1-2*(xx+zz),   2*(yz-xw)], 
             [  2*(xz-yw),   2*(yz+xw), 1-2*(xx+yy)]]
        )

        return rotation_matrix


    def qrotate(self, v, q):
        """
        Rotate vector(s) v about the rotation described by quaternion(s) q.
        Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
        where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).
        """
        original_shape = list(v.shape)

        q = torch.stack([q] * original_shape[0])
        # q = torch.stack([q] * self.num_features)

        q = q.view(-1, 4)
        v = v.view(-1, 3)

        qvec = q[:, 1:]
        uv = torch.cross(qvec, v, dim=1)
        uuv = torch.cross(qvec, uv, dim=1)
        return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


    def qmul(self, q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4
        
        original_shape = q.shape
        
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)


    def inverse_rotation_quaternion(self, q): 
        original_shape = q.shape
        q = q.view(4)
        q_inv = torch.zeros(4)
        q_inv[0] = q[0]
        for i in range(3):
            q_inv[i+1] = -q[i+1]
        q_inv = q_inv.view(original_shape)
        return q_inv


    #################################################################################
    #################### TRANSLATION
    #################################################################################


    def init_translation_bias_(self):
        # tb = torch.rand(self.dimensions)
        tb = torch.ones(self.dimensions)
        return tb


    def update_translation_bias_(self, translation_bias, gradient, learning_rate):
        upd_translation_bias = Variable(
            translation_bias - learning_rate * gradient,
            requires_grad=self.translation_gradient_init)

        return upd_translation_bias
    
    def update_translation_bias_(self, translation_bias, gradient, learning_rate, momentum):
        mom = momentum*self.calc_momentum_translation()
        
        upd_translation_bias = Variable(
            translation_bias - learning_rate * gradient + mom,
            requires_grad=self.translation_gradient_init)
        # print('Updated translation bias.')

        self.update_momentum_translation(translation_bias)

        return upd_translation_bias


    def translate(self, input, translation_bias): 
        return input + translation_bias

    
    def update_momentum_translation(self, entries): 
        if self.bin_momentum_translation[0] == None: 
            self.bin_momentum_translation[0] = torch.Tensor(entries.clone().detach())
            self.bin_momentum_translation[1] = torch.Tensor(entries.clone().detach())
        else: 
            self.bin_momentum_translation[1] = self.bin_momentum_translation[0]
            self.bin_momentum_translation[0] = torch.Tensor(entries.clone().detach())
    
    def calc_momentum_translation(self):
        if self.bin_momentum_translation[0] == None: 
            return torch.zeros(self.dimensions)
        else:
            return self.bin_momentum_translation[1] - self.bin_momentum_translation[0]
    

    def inverse_translation_bias(self, t): 
        return t * -1


def main(): 
    perspective_taker = Perspective_Taker(15, 3, rotation_gradient_init=True, translation_gradient_init=True)
    
    rm = perspective_taker.compute_rotation_matrix_(torch.tensor([0.]), torch.tensor([90.]), torch.tensor([180.]))
    print(rm)

if __name__ == "__main__":
    main()
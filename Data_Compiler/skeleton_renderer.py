import pygame
import os 
import torch


class SKEL_RENDERER():

    def __init__(self):
        os.environ["SDL_VIDEO_CENTERED"]='1'
        self.black, self.white, self.red, self.blue, self.green = (0,0,0), (250,250,250), (255,0,0), (0,0,255), (0,255,0)
        self.width, self.height = 2000, 2000


    def connection(self, j1, j2, skeleton): 
        p1 = skeleton[j1]
        p2 = skeleton[j2]
        pygame.draw.line(
            self.screen, 
            self.black, 
            (p1[0], p1[1]), 
            (p2[0], p2[1]), 
            2)


    def direction_arrow(self, joint, dir_pont, mag): 
        pygame.draw.line(
            self.screen, 
            self.green, 
            (joint[0], joint[1]), 
            (dir_pont[0], dir_pont[1]),
            mag
            )
        # pygame.draw.polygon()



    def render(self, position, direction, magnitude, gestalt):

        pygame.init()
        pygame.display.set_caption("3D skeleton")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.fps = 100

        run = True
        frame_cnt = 0
        num_frames = position.size()[0]
        num_features = position.size()[1]
        up_scale = 500

        while run: 
            self.clock.tick(self.fps)

            c_frame_pos = position[frame_cnt] *up_scale
            if gestalt:
                # c_frame_dir = direction[frame_cnt] *up_scale
                c_frame_mag = magnitude[frame_cnt]
                c_frame_dir = direction[frame_cnt] *up_scale

            self.screen.fill(self.white)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
            center = (1000,1000)
            proj_skeleton = torch.tensor([])
            for i in range(num_features):
                joint = c_frame_pos[i]
                if gestalt:
                    mag_arrow = c_frame_mag[i]
                    dir_arrow = c_frame_dir[i] *mag_arrow *10

                z = 1
                projection_matrix = torch.Tensor([
                    [z,0,z], 
                    [0,-z,0]
                ])
                
                projected_joint = torch.matmul(projection_matrix, joint)
                x_j = int(projected_joint[0] + center[0])
                y_j = int(projected_joint[1] + center[1])
                projected_joint = torch.tensor([x_j, y_j])

                if gestalt:
                    dir_point = torch.matmul(projection_matrix, dir_arrow)
                    x_d = int(projected_joint[0] + dir_point[0])
                    y_d = int(projected_joint[1] + dir_point[1])
                    dir_point = torch.tensor([x_d, y_d])

                    self.direction_arrow(projected_joint, dir_point, 5)
                    # self.direction_arrow(projected_joint, dir_point, mag_arrow)
                    # pygame.draw.circle(self.screen, self.red, (x_d, y_d), 10)
                    # pygame.draw.circle(self.screen, self.red, (x_d, y_d), mag_arrow)

                pygame.draw.circle(self.screen, self.blue, (x_j, y_j), 10)
                proj_skeleton = torch.cat([proj_skeleton, projected_joint.view(1,2)])
            
            proj_skeleton = proj_skeleton.int()

            if num_features == 15: 
                self.connection(0,1, proj_skeleton)
                self.connection(1,2, proj_skeleton)
                self.connection(3,4, proj_skeleton)
                self.connection(4,5, proj_skeleton)

                self.connection(0,6, proj_skeleton)
                self.connection(3,6, proj_skeleton)

                self.connection(6,7, proj_skeleton)
                self.connection(7,8, proj_skeleton)

                self.connection(7,9, proj_skeleton)
                self.connection(9,10, proj_skeleton)
                self.connection(10,11, proj_skeleton)
                self.connection(7,12, proj_skeleton)
                self.connection(12,13, proj_skeleton)
                self.connection(13,14, proj_skeleton)

            elif num_features >= 16: 
                self.connection(0,1, proj_skeleton)
                self.connection(1,2, proj_skeleton)
                self.connection(3,4, proj_skeleton)
                self.connection(4,5, proj_skeleton)

                self.connection(0,6, proj_skeleton)
                self.connection(3,6, proj_skeleton)

                self.connection(6,7, proj_skeleton)
                self.connection(7,9, proj_skeleton)

                self.connection(7,10, proj_skeleton)
                self.connection(10,11, proj_skeleton)
                self.connection(11,12, proj_skeleton)
                self.connection(7,13, proj_skeleton)
                self.connection(13,14, proj_skeleton)
                self.connection(14,15, proj_skeleton)

            

            if frame_cnt<num_frames-1: frame_cnt += 1
            else: run = False

            pygame.display.update()

        pygame.quit()


    


def main(): 
    skelrenderer = SKEL_RENDERER()
    # data = torch.load("BA_BAPTAT/Grafics/CombinedBindingRuns/combination_t_b_r_gest_parameter_settings/2021_Mar_25-18_28_07/b_r_t_num_tuning_cycles_2/S35T07/"+"final_predictions.pt")
    # data = torch.load("Grafics/CombinedBindingRuns/combination_t_b_r_gest_parameter_settings/2021_Mar_27-17_03_09-decay-dgrad/b_r_t_num_tuning_cycles_3/S35T07/"+"final_predictions.pt")
    # data = torch.load("Grafics/CombinedBindingRuns/combination_t_b_r_gest_parameter_settings/2021_Mar_27-17_03_09-decay-dgrad/b_r_t_num_tuning_cycles_3/S35T07/"+"final_inputs.pt")
    # data = torch.load('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/GestaltRuns/compare_gestalten_rotation/2021_Mar_29-20_31_47/r_dimensions_6/S35T07_qrotated/final_inputs.pt')
    data = torch.load('D:/Uni/Kogni/Bachelorarbeit/Code/BA_BAPTAT/Grafics/GestaltRuns/compare_gestalten_rotation/2021_Mar_29-20_31_47/r_dimensions_7/S35T07/final_inputs.pt')
    # data = torch.load("BA_BAPTAT/Grafics/SeparateBindingRuns/binding_sigmoid_rw_cw_rcw_gest/2021_Mar_26-14_45_05/rcwSM/S35T07_rebinded")
    # data = torch.load("BA_BAPTAT/Grafics/SeparateRotationRuns/rotation_euler_vs_quaternion/saveruns/2021_Mar_26-09_24_18/rotationType_qrotate/S35T07_qrotated/final_inputs")
    # data = torch.load("BA_BAPTAT/Grafics/SeparateRotationRuns/rotation_euler_vs_quaternion/saveruns/2021_Mar_26-09_24_18/rotationType_qrotate/S35T07_qrotated/final_predictions")
    gestalt = True
    print(data.shape)

    num_frames = 991
    num_features = 15
    num_dimensions = 7

    if gestalt:
        data = data.view(num_frames-1, num_features, num_dimensions)
        pos = data[:,:,:3]
        dir = data[:,:,3:6]
        mag = data[:,:,-1]

        skelrenderer.render(pos, dir, mag, gestalt)
    else:
        data = data.view(num_frames, num_features, num_dimensions)
        skelrenderer.render(data, None, None, gestalt)


if __name__ == "__main__":
    main()

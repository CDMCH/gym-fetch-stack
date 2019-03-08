import gym_fetch_stack
import gym
import pygame
import numpy as np
# import cv2

# On an Xbox One Controller (On Linux) The controls are:
# Right Stick - control horizontal movement
# Bumpers - control vertical movement
# Left Stick (up/down) - close/open gripper


if __name__ == '__main__':

    from gym import envs

    all_envs = envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    print(env_ids)

    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    env = gym.make('FetchStack6Sparse-v1')
#    env = gym.make('FetchStack2TrainerOneThirdIsStacking-v1')
    env.reset()

    print("env action space high: {}".format(env.action_space.high))
    print("env action space low: {}".format(env.action_space.low))
    print("env _max_episode_steps: {}".format(env._max_episode_steps))

    frames = 0

    g = 0

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 24.0, (2560, 1440))
    # recording = False

    while True:

        # action = env.action_space.sample()
        pygame.event.get()
        x = joystick.get_axis(4) * 0.5
        y = joystick.get_axis(3) * 0.5

        if joystick.get_button(0):
            env.reset()
            continue

        if joystick.get_button(4):
            z = -0.5
        elif joystick.get_button(5):
            z = 0.5
        else:
            z = 0

        g_adjust = joystick.get_axis(1)
        if abs(g_adjust) > 0.6:
            g = max(-1, min(1, g + (g_adjust/abs(g_adjust) * 0.04)))
        # for i in range(6):
        #     print('{}:{} '.format(i, joystick.get_button(i), end=' '))

        # z = joystick.get_axis(1) * -1

        deadzone = 0.2

        if abs(x) < deadzone:
            x = 0
        if abs(y) < deadzone:
            y = 0

        action = np.asarray([x, y, z, g])

        # print(action)

        frame = env.render('rgb_array')[...,::-1]
        # cv2.imshow("robot",frame)
        # if ord('r'):
        #     recording = True
        # if recording:
        #     out.write(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


        step_results = env.step(action)
        obs, reward, done, info = step_results
        # print("Reward: {} Info: {}".format(reward, info))
        # print("OBS: {} | REWARD: {} | DONE: {} | INFO: {}".format(obs, reward, done, info))

        # if done or info['is_success']:
        #     env.reset()

        frames += 1

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # env.close()
    # print("done")

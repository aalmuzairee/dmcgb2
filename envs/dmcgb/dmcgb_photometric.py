# Adapted from DMC-GB benchmark: https://github.com/nicklashansen/dmcontrol-generalization-benchmark/


import os
import torch
import numpy as np
import cv2
import xmltodict
import copy
import collections
from PIL import Image


import dm_env
import dm_control
import dm_control.suite.wrappers
from dm_control.mujoco.wrapper import mjbindings



dmcgb_photometric_modes = ['color_easy', 'color_hard', 'video_easy', 'video_hard', 'color_video_easy', 'color_video_hard']
dmcgb_data_dir = "envs/dmcgb/data"


class ColorVideoWrapper(dm_env.Environment):
    ''' DMCGB Wrapper for dmcontrol suite for applying changes in colors and videos, must be applied before pixel wrapper'''
    def __init__(self, env, mode, seed, video_render_size=256):
        self._env = env
        self._mode = mode
        self._seed = seed
        self._random_state = np.random.RandomState(seed)
        self._video_render_size = video_render_size

        # XML of current domain
        self._xml = self._get_model_and_assets(self._env._domain_name+'.xml')
        
        # Video
        self._video_paths = []
        self._current_video_frame = 0 # Which frame in video, placeholder
        self._current_video_len = 1 # Length of video, placeholder
        self._SKY_TEXTURE_INDEX = 2 # Default skybox
        self._Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))

        # Mode
        self._color_in_effect= 'color' in self._mode
        self._video_in_effect= 'video' in self._mode
        self._remove_ground_and_rails = (self._mode == 'video_hard') 
        self._moving_domain = self._env._domain_name in ['walker', 'cheetah'] # Background needs to move with them
        self._moving_domain_offset_x = 0 if self._env._domain_name == 'walker' else -0.05 # Walker or Cheetah 
        self._moving_domain_offset_z = -1.07 if self._env._domain_name == 'walker' else 0.15

        # Loading 
        start_index = self._random_state.randint(100)
        if self._color_in_effect:
            self._load_colors() # Get Colors
            self._num_colors = len(self._colors)
            assert self._num_colors >= 100, 'env must include at least 100 colors'
            self._color_index =  start_index % self._num_colors
        if self._video_in_effect:
            self._get_video_paths() # Get Videos
            self._num_videos = len(self._video_paths)
            self._video_index = start_index % self._num_videos
            self._reload_physics(*self._reformat_xml({})) # Create backcube with video




# -------------------Video Helpers--------------------------------------------

    def _get_video_paths(self):
        if 'easy' in self._mode:
            video_dir = os.path.join(dmcgb_data_dir, 'video_easy')
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
        elif 'hard' in self._mode:
            video_dir = os.path.join(dmcgb_data_dir, 'video_hard')
            self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
        else:
            raise ValueError(f'received unknown mode "{self._mode}"')

    def _load_video(self, video):
        """Load video from provided filepath and return as numpy array"""
        cap = cv2.VideoCapture(video)
        assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
        assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
        i, ret = 0, True
        while (i < n  and ret):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf[i] = frame
            i += 1
        cap.release()
        return buf


    # This function forces the wrapper to be before the environment is rendered in the pixel wrapper
    def _move_backcube(self): 
        """ Moves the backcube to follow camera frame when dealing with moving domains like cheetah and walker """
        if self._moving_domain:
            body_x_pos = self._env.physics.data.body('torso').subtree_com[0]
            body_z_pos = self._env.physics.data.body('torso').subtree_com[2]
            self._env.physics.data.site('video_screen').xpos[0] = body_x_pos + self._moving_domain_offset_x
            self._env.physics.data.site('video_screen').xpos[2] = body_z_pos + self._moving_domain_offset_z


    def _reset_background(self, render_size=None):
        """ Sets the stage for video background in simulation and loads and prepares the video images """
        # Extra things to remove
        if self._remove_ground_and_rails:
            self._env.physics.named.model.mat_rgba['grid', 'a'] = 0 # Removing grid

        # Set image size in simulation
        if render_size is not None:
            self._video_render_size = render_size
        sky_height = self._env.physics.model.tex_height[self._SKY_TEXTURE_INDEX] = sky_width = self._env.physics.model.tex_width[self._SKY_TEXTURE_INDEX]=self._video_render_size
        sky_size = sky_height * sky_width * 3
        sky_address = self._env.physics.model.tex_adr[self._SKY_TEXTURE_INDEX]
        # Load images from video
        self._video_index = (self._video_index + 1) % self._num_videos
        images = self._load_video(self._video_paths[self._video_index])
        self._current_video_len = len(images)
        # Generate image textures
        texturized_images = []
        for image in images:
            image_flattened = self._size_and_flatten(image, sky_height, sky_width)
            texturized_images.append(image_flattened)
        self._background = self._Texture(sky_size, sky_address, texturized_images)


    def _apply_video(self):
        """Apply the background video texture to the backcube and increment counter"""
        assert self._background is not None, "Missing reference to skybox background in VideoWrapper"
        start = self._background.address
        end = self._background.address + self._background.size
        texture = self._background.textures[self._current_video_frame]
        self._env.physics.model.tex_rgb[start:end] = texture
        # Upload the new texture to the GPU. 
        with self._env.physics.contexts.gl.make_current() as ctx:
            ctx.call(
                mjbindings.mjlib.mjr_uploadTexture,
                self._env.physics.model.ptr,
                self._env.physics.contexts.mujoco.ptr,
                self._SKY_TEXTURE_INDEX,
            )
        # Increment
        self._current_video_frame = (self._current_video_frame + 1) % self._current_video_len
    

    def _size_and_flatten(self, image, ref_height, ref_width):
        """ Resize image if necessary and flatten the result """
        image_height, image_width = image.shape[:2]
        if image_height != ref_height or image_width != ref_width:
            image = np.asarray(Image.fromarray(image).resize(size=(ref_width, ref_height)))
        return image.flatten(order='K')


    def _render_high(self, size=256, camera_id=0):
        """ 
        Utility function to override original set background video resolution with a higher resolution for recording
        This function changes the background videos resolution for the environment, which will slow the speed of the env.
        """
        if size != self._video_render_size:
            self._reset_background(render_size=size)
            self._video_render_size = size
        return self._env.physics.render(height=size, width=size, camera_id=camera_id)

# -------------------------------Color helpers------------------------------------

    def _load_colors(self):
        if 'hard' in self._mode: 
            self._colors = torch.load(f'{dmcgb_data_dir}/color_hard.pt')
        elif 'easy' in self._mode:
            self._colors = torch.load(f'{dmcgb_data_dir}/color_easy.pt')

    def _randomize_colors(self):
        chosen_colors =  self._colors[self._color_index]
        self._reload_physics(*self._reformat_xml(chosen_colors))
        self._color_index = (self._color_index + 1) % self._num_colors
        

    def _reload_physics(self, xml_string, assets=None):
        assert hasattr(self._env, 'physics'), 'environment does not have physics attribute'
        # For newer mujoco need to convert from str to bytes
        if assets:
            new_assets = {}
            for key, val in assets.items():
                if type(val) == bytes:
                    new_assets[key] = val
                else:
                    new_assets[key] = val.encode('utf-8')
            assets = new_assets
        self._env.physics.reload_from_xml_string(xml_string, assets=assets)

    def _get_model_and_assets(self, model_fname):
        """"Returns a tuple containing the model XML string and a dict of assets."""
        # ball_in_cup different name
        if model_fname == "cup.xml":
            model_fname = "ball_in_cup.xml"
        # Convert XML to dicts
        model = dm_control.suite.common.read_model(model_fname)
        assets = dm_control.suite.common.ASSETS
        return model, assets


    def _reformat_xml(self, chosen_colors):
        model_xml, assets = self._xml
        model_xml = copy.deepcopy(model_xml)
        assets = copy.deepcopy(assets)

        # Convert XML to dicts
        model = xmltodict.parse(model_xml)
        materials = xmltodict.parse(assets['./common/materials.xml'])
        skybox = xmltodict.parse(assets['./common/skybox.xml'])

        # Edit grid floor
        if 'grid_rgb1' in chosen_colors:
            assert isinstance(chosen_colors['grid_rgb1'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@rgb1'] = \
                f'{chosen_colors["grid_rgb1"][0]} {chosen_colors["grid_rgb1"][1]} {chosen_colors["grid_rgb1"][2]}'
        if 'grid_rgb2' in chosen_colors:
            assert isinstance(chosen_colors['grid_rgb2'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@rgb2'] = \
                f'{chosen_colors["grid_rgb2"][0]} {chosen_colors["grid_rgb2"][1]} {chosen_colors["grid_rgb2"][2]}'
        if 'grid_markrgb' in chosen_colors:
            assert isinstance(chosen_colors['grid_markrgb'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['texture']['@markrgb'] = \
                f'{chosen_colors["grid_markrgb"][0]} {chosen_colors["grid_markrgb"][1]} {chosen_colors["grid_markrgb"][2]}'
        if 'grid_texrepeat' in chosen_colors:
            assert isinstance(chosen_colors['grid_texrepeat'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['texture']['@name'] == 'grid'
            materials['mujoco']['asset']['material'][0]['@texrepeat'] = \
                f'{chosen_colors["grid_texrepeat"][0]} {chosen_colors["grid_texrepeat"][1]}'

        # Edit self
        if 'self_rgb' in chosen_colors:
            assert isinstance(chosen_colors['self_rgb'], (list, tuple, np.ndarray))
            assert materials['mujoco']['asset']['material'][1]['@name'] == 'self'
            materials['mujoco']['asset']['material'][1]['@rgba'] = \
                f'{chosen_colors["self_rgb"][0]} {chosen_colors["self_rgb"][1]} {chosen_colors["self_rgb"][2]} 1'

        # Edit skybox
        if 'skybox_rgb' in chosen_colors:
            assert isinstance(chosen_colors['skybox_rgb'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@rgb1'] = \
                f'{chosen_colors["skybox_rgb"][0]} {chosen_colors["skybox_rgb"][1]} {chosen_colors["skybox_rgb"][2]}'
        if 'skybox_rgb2' in chosen_colors:
            assert isinstance(chosen_colors['skybox_rgb2'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@rgb2'] = \
                f'{chosen_colors["skybox_rgb2"][0]} {chosen_colors["skybox_rgb2"][1]} {chosen_colors["skybox_rgb2"][2]}'
        if 'skybox_markrgb' in chosen_colors:
            assert isinstance(chosen_colors['skybox_markrgb'], (list, tuple, np.ndarray))
            assert skybox['mujoco']['asset']['texture']['@name'] == 'skybox'
            skybox['mujoco']['asset']['texture']['@markrgb'] = \
                f'{chosen_colors["skybox_markrgb"][0]} {chosen_colors["skybox_markrgb"][1]} {chosen_colors["skybox_markrgb"][2]}'

        # For Videos Add a Cube/Box Behind Model to Project Videos on
        if self._video_in_effect:
            domain = self._env._domain_name

            # Adding texture to update with videos
            materials['mujoco']['asset']['texture'] = [ materials['mujoco']['asset']['texture'],
                                                       {'@name':'projector', '@type':'skybox', '@builtin':'flat', '@width':'512', '@height':'512', '@mark':'none'}]
            materials['mujoco']['asset']['material'].append({'@name':'projector', '@texture':'projector', '@texuniform':'false', '@specular':'0', '@shininess':'0', '@reflectance':'0', '@emission':'0'})

            # Projector Cubes to add as sites
            site_dicts = {
                'walker': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 1.96 -0.7', '@euler': '-30 0 0', '@material':'projector'},
                'cheetah': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 1.6 -0.7', '@euler': '0 0 0', '@material':'projector'},
                'cartpole': {'@name':'video_screen', '@type':'box', '@size':'1.86 0.1 1.86', '@pos':'0.0 0.5 1.0', '@euler': '0 0 0', '@material':'projector'},
                'ball_in_cup': {'@name':'video_screen', '@type':'box', '@size':'0.66 0.1 0.66', '@pos':'0.0 0.5 0.05', '@euler': '-27 0 0', '@material':'projector'},
                'finger': {'@name':'video_screen', '@type':'box', '@size':'0.66 0.1 0.66', '@pos':'0.0 0.5 0.05', '@euler': '-27 0 0', '@material':'projector'},
            }

            # Adding sites to xmls
            if domain in ['walker', 'cheetah', 'cartpole', 'ball_in_cup']:
                model['mujoco']['worldbody']['site'] = site_dicts[domain]
            elif domain in ['finger']:
                model['mujoco']['worldbody']['site'] = [model['mujoco']['worldbody']['site'], site_dicts[domain]]


        # Convert back to XML
        model_xml = xmltodict.unparse(model)
        assets['./common/materials.xml'] = xmltodict.unparse(materials)
        assets['./common/skybox.xml'] = xmltodict.unparse(skybox)

        return model_xml, assets


# --------------------------------Main functions--------------------------------
    def reset(self):
        """Reset the background state."""
        if self._color_in_effect: # loads a color and resets env with xml
            self._randomize_colors()
        time_step = self._env.reset()
        if self._video_in_effect: # removes backgrounds and updates textures/backcube
            self._current_video_frame = 0
            self._reset_background()
            self._apply_video()
            self._move_backcube()
        return time_step
    
    def step(self, action):
        time_step = self._env.step(action)
        if self._video_in_effect:
            self._apply_video()
            self._move_backcube()
        return time_step

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)














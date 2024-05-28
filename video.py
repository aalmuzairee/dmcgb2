
import numpy as np
import os
import moviepy.editor as mp


class VideoRecorder:
    def __init__(self, save_video=False, video_dir=None, render_size=256, fps=30):
        self.save_video = save_video
        self.video_dir = video_dir
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        
        if self.video_dir:
            try: 
                os.makedirs(self.video_dir, exist_ok = True) 
            except OSError as error: 
                self.video_dir = None
                pass

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = enabled
        self.record(env)

    def record(self, env):
        if self.enabled and self.save_video:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)
    

    def save(self, file_name, wandb):
        if self.enabled and self.save_video:
            if self.video_dir:
                path = str(self.video_dir) +  f"/{file_name}.mp4"
                # Using moviepy
                clip = mp.ImageSequenceClip(self.frames, fps=self.fps)
                clip.write_videofile(path, verbose=False, logger=None)
            if wandb:
                frames = np.stack(self.frames).transpose(0, 3, 1, 2)
                wandb.log({file_name: wandb.Video(frames, fps=self.fps, format='mp4')})






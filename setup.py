from setuptools import setup


setup(name='gym_fetch_stack',
      version=1.0,
      description='OpenAI Gym Fetch Stack, modified from the OpenAI robotics Mujoco fetch environments',
      zip_safe=False,
      install_requires=[
          'numpy', 'gym', 'mujoco_py>=1.50' 'imageio'
      ],
      package_data={'gym': [
        'assets/LICENSE.md',
        'assets/fetch/*.xml',
        'assets/stls/fetch/*.stl',
        'assets/textures/*.png']
      }
)

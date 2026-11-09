import os
from typing import Optional
from pydantic_settings import BaseSettings

def sanitize_directory_name(dir_name):
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        dir_name = dir_name.replace(char, '')
    dir_name = ''.join(c for c in dir_name if c >= ' ')  # remove characters with ASCII value less than 32
    dir_name = dir_name.replace(' ', '_')
    # replace 'ä' with 'ae', 'ö' with 'oe', 'ü' with 'ue'
    dir_name = dir_name.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    # replace 'Ä' with 'Ae', 'Ö' with 'Oe', 'Ü' with 'Ue'
    dir_name = dir_name.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    # replace 'ß' with 'ss'
    dir_name = dir_name.replace('ß', 'ss')

    return dir_name

# define the settings of the inference server
class Settings(BaseSettings):

    # self._env_file = "inf_server_config.env"

    VERSION: str = "1.0"
    SERVICE_PORT: int = 8001
    
    MINIO_USER: str = ""
    MINIO_PASSWORD: str = ""

    S3_USER: str = ""
    S3_PASSWORD: str = ""

    CVAT_HOST: str = "localhost"
    CVAT_PORT: int = 8080
    CVAT_USER: str = ""
    CVAT_PASSWORD: str = ""

    VIDEO_BASE_URL: str = ""  # http://localhost:8085/videos"
    CALLBACK_COMPLETED_URL: str = ""  # "http://localhost:8085/callback_finished"
    MAX_NUMB_WAITING_JOBS: int = 10
    CONFIG_FILE: Optional[str] = ''
    MODEL_TYPE: Optional[str] = 'dinov3'
    MODEL_SIZE: Optional[str] = 'l'
    PROTOTYPE_DIR: Optional[str] = 'prototypes'
    DETECTOR_TYPE: Optional[str] = 'open-vocabulary'
    BASE_TRAIN_DATASET: Optional[str] = 'lvis'
    OBJECT_DETECTION_THRESHOLD: Optional[float] = 0.1
    IOA_THRESHOLD: Optional[float] = 0.0
    MIN_REGION_AREA: Optional[int] = 0
    SEGMENT_REGION_FROM_CLASS_DETECTION: Optional[bool] = False
    TOP_K: Optional[int] = 1
    TOP_K_FOR_FINE_TUNING: Optional[int] = 7
    MASK_MODE: Optional[bool] = False
    WEIGHTED_PATCH: Optional[bool] = False
    CHECK_MASK_OVERLAP: Optional[bool] = True

    PROCESSING_DIR: Optional[str] = './temp'
    TRAINING_DIR: Optional[str] = './web_gui_demo_data/dataset/training'
    VALIDATION_DIR: Optional[str] = './web_gui_demo_data/dataset/validation'
    DELETE_PREV_JOB_DATA: Optional[bool] = True

    def __init__(self, _env_file: str = "server_config.env"):
        super().__init__(_env_file=_env_file)
        # check if environment variable are set for S3 access settings
        if 'S3_USER' in os.environ:
            self.S3_USER = os.environ['S3_USER']
        if 'S3_PASSWORD' in os.environ:
            self.S3_PASSWORD = os.environ['S3_PASSWORD']
        if 'MINIO_USER' in os.environ:
            self.MINIO_USER = os.environ['MINIO_USER']
        if 'MINIO_PASSWORD' in os.environ:
            self.MINIO_PASSWORD = os.environ['MINIO_PASSWORD']
        if 'CVAT_HOST' in os.environ:
            self.CVAT_HOST = os.environ['CVAT_HOST']
            print(f"CVAT_HOST environment variable: {self.CVAT_HOST}")
        else:
            print('CVAT_HOST not set in environment variables!')
        if 'CVAT_PORT' in os.environ:
            self.CVAT_PORT = os.environ['CVAT_PORT']
        if 'CVAT_USER' in os.environ:
            self.CVAT_USER = os.environ['CVAT_USER']
        if 'CVAT_PASSWORD' in os.environ:
            self.CVAT_PASSWORD = os.environ['CVAT_PASSWORD']


    def set_from_json(self, new_settings):
        if 'VERSION' in new_settings:
            self.VERSION = new_settings['VERSION']
        if 'SERVICE_PORT' in new_settings:
            self.SERVICE_PORT = new_settings['SERVICE_PORT']
        if 'MINIO_USER' in new_settings:
            self.MINIO_USER = new_settings['MINIO_USER']
        if 'MINIO_PASSWORD' in new_settings:
            self.MINIO_PASSWORD = new_settings['MINIO_PASSWORD']
        if 'S3_USER' in new_settings:
            self.S3_USER = new_settings['S3_USER']
        if 'S3_PASSWORD' in new_settings:
            self.S3_PASSWORD = new_settings['S3_PASSWORD']
        if 'CVAT_HOST' in new_settings:
            self.CVAT_HOST = new_settings['CVAT_HOST']
        if 'CVAT_PORT' in new_settings:
            self.CVAT_PORT = new_settings['CVAT_PORT']
        if 'CVAT_USER' in new_settings:
            self.CVAT_USER = new_settings['CVAT_USER']
        if 'CVAT_PASSWORD' in new_settings:
            self.CVAT_PASSWORD = new_settings['CVAT_PASSWORD']
        if 'PROCESSING_DIR' in new_settings:
            self.PROCESSING_DIR = new_settings['PROCESSING_DIR']
        if 'VIDEO_BASE_URL' in new_settings:
            self.VIDEO_BASE_URL = new_settings['VIDEO_BASE_URL']
        if 'MODEL_SIZE' in new_settings:
            self.MODEL_SIZE = new_settings['MODEL_SIZE']
        if 'PROTOTYPE_DIR' in new_settings:
            self.PROTOTYPE_DIR = new_settings['PROTOTYPE_DIR']
        if 'DETECTOR_TYPE' in new_settings:
            self.DETECTOR_TYPE = new_settings['DETECTOR_TYPE']
        if 'BASE_TRAIN_DATASET' in new_settings:
            self.BASE_TRAIN_DATASET = new_settings['BASE_TRAIN_DATASET']
        if 'OBJECT_DETECTION_THRESHOLD' in new_settings:
            self.OBJECT_DETECTION_THRESHOLD = new_settings['OBJECT_DETECTION_THRESHOLD']
        if 'OVERLAPPING_MODE' in new_settings:
            self.OVERLAPPING_MODE = new_settings['OVERLAPPING_MODE']
        if 'MIN_REGION_AREA'  in new_settings:
            self.MIN_REGION_AREA = new_settings['MIN_REGION_AREA']
        if 'SEGMENT_REGION_FROM_CLASS_DETECTION' in new_settings:
            self.SEGMENT_REGION_FROM_CLASS_DETECTION = new_settings['SEGMENT_REGION_FROM_CLASS_DETECTION']
        if 'TOP_K' in new_settings:
            self.TOP_K = new_settings['TOP_K']
        if 'MASK_MODE' in new_settings:
            self.MASK_MODE = new_settings['MASK_MODE']
        if 'WEIGHTED_PATCH' in new_settings:
            self.WEIGHTED_PATCH = new_settings['WEIGHTED_PATCH']
        if 'CHECK_MASK_OVERLAP' in new_settings:
            self.CHECK_MASK_OVERLAP = new_settings['CHECK_MASK_OVERLAP']
        if 'CALLBACK_COMPLETED_URL' in new_settings:
            self.CALLBACK_COMPLETED_URL = new_settings['CALLBACK_COMPLETED_URL']
        if 'MAX_NUMB_WAITING_JOBS' in new_settings:
            self.MAX_NUMB_WAITING_JOBS = new_settings['MAX_NUMB_WAITING_JOBS']
        if 'DELETE_PREV_JOB_DATA' in new_settings:
            self.DELETE_PREV_JOB_DATA = new_settings['DELETE_PREV_JOB_DATA']


    def get_json(self):
        return {
            'VERSION': self.VERSION,
            'SERVICE_PORT': self.SERVICE_PORT,
            'MINIO_USER': self.MINIO_USER,
            'S3_USER': self.S3_USER,
            'MAX_NUMB_WAITING_JOBS': self.MAX_NUMB_WAITING_JOBS,
            'OBJECT_DETECTION_THRESHOLD': self.OBJECT_DETECTION_THRESHOLD
        }

    # save settings to file
    def save(self):
        with open(self._env_file, 'w') as f:
            f.write(f"VERSION={self.VERSION}\n")
            f.write(f"SERVICE_PORT={self.SERVICE_PORT}\n")
            f.write(f"MINIO_USER={self.MINIO_USER}\n")
            f.write(f"MINIO_PASSWORD={self.MINIO_PASSWORD}\n")
            f.write(f"S3_USER={self.S3_USER}\n")
            f.write(f"S3_PASSWORD={self.S3_PASSWORD}\n")
            f.write(f"CVAT_HOST={self.CVAT_HOST}\n")
            f.write(f"CVAT_PORT={self.CVAT_PORT}\n")
            f.write(f"CVAT_USER={self.CVAT_USER}\n")
            f.write(f"CVAT_PASSWORD={self.CVAT_PASSWORD}\n")
            f.write(f"VIDEO_BASE_URL={self.VIDEO_BASE_URL}\n")
            f.write(f"CALLBACK_COMPLETED_URL={self.CALLBACK_COMPLETED_URL}\n")
            f.write(f"MAX_NUMB_WAITING_JOBS={self.MAX_NUMB_WAITING_JOBS}\n")
            f.write(f"MODEL_SIZE={self.MODEL_SIZE}\n")
            f.write(f"PROTOTYPE_DIR={self.PROTOTYPE_DIR}\n")
            f.write(f"DETECTOR_TYPE={self.DETECTOR_TYPE}\n")
            f.write(f"BASE_TRAIN_DATASET={self.BASE_TRAIN_DATASET}\n")
            f.write(f"OBJECT_DETECTION_THRESHOLD={self.OBJECT_DETECTION_THRESHOLD}\n")
            f.write(f"OVERLAPPING_MODE={self.OVERLAPPING_MODE}\n")

class Path(object):
    @staticmethod
    def pretrained_c3d():
        return 'python/video_classification/pretrained/c3d-pretrained.pth'
    
    @staticmethod
    def ucf_labels():
        return 'python/video_classification/dataloaders/ucf_labels.txt'
    
    @staticmethod
    def hmdb_labels():
        return 'python/video_classification/dataloaders/hmdb_labels.txt'
    
    @staticmethod
    def video():
        return '/mnt/data/UCF-101/Rowing/v_Rowing_g01_c02.avi'
    
    @staticmethod
    def perturbed_video():
        return 'python/noise_perturbation/perturbed_video.avi'
    
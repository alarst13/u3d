class Path(object):
    @staticmethod
    def pretrained_c3d():
        return 'video_classification/pretrained/c3d-pretrained.pth'
    
    @staticmethod
    def video():
        return '/mnt/data/UCF-101/Rowing/v_Rowing_g01_c02.avi'
    
    @staticmethod
    def perturbed_video():
        return 'noise_perturbation/perturbed_video.avi'
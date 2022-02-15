import torch
from conv_stft import ConvSTFT


class stftm_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.stft =  ConvSTFT(frame_size, frame_shift, frame_size, 'hanning', 'complex', fix=True).cuda()
        self.fft_len = 512


    def __call__(self, outputs, labels):
        out_real, out_imag = self.get_stftm(outputs)
        lab_real, lab_imag = self.get_stftm(labels)

        if self.loss_type == 'mae':
            loss = torch.mean(torch.abs(out_real-lab_real)+torch.abs(out_imag-lab_imag))
        elif self.loss_type == 'char':
            loss =  self.char_loss(out_real, lab_real) + self.char_loss(out_imag, lab_imag)
        elif self.loss_type == 'hybrid':
            loss = (self.edge_loss(out_real, lab_real) + self.edge_loss(out_imag, lab_imag)) * 0.05 +\
                    self.char_loss(out_real, lab_real) + self.char_loss(out_imag, lab_imag)


        return loss


    def get_stftm(self, ipt):
        specs = self.stft(ipt)

        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]

        return real, imag



class mag_loss(object):
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        self.stft =  ConvSTFT(frame_size, frame_shift, frame_size, 'hanning', 'complex', fix=True).cuda()
        self.fft_len = 512

    def __call__(self, outputs, labels):
        out_mags = self.get_mag(outputs)
        lab_mags = self.get_mag(labels)

        if self.loss_type == 'mae':
            loss = torch.mean(torch.abs(out_mags-lab_mags))

        return loss
    
    def get_mag(self, ipt):

        specs = self.stft(ipt)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec_mags = torch.sqrt(real**2+imag**2+1e-8)

        return spec_mags


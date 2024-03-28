# from scipy.fftpack import dct, idct
import torch
import numpy as np
import torch.nn.functional as F
from scipy.signal import kaiser

def mdct_basis_(N):
    n0 = ((N//2) + 1) /2
    idx   = np.arange(0,N,1).reshape(N, 1)  
    kn    = np.multiply(idx + n0,(idx[:(N//2),:] + 0.5).T)
    basis = np.cos((2*np.pi/N)*kn)
    return torch.FloatTensor(basis.T)

def kbd_window_(win_len, filt_len, alpha=4):
    window = np.cumsum(kaiser(int(win_len/2)+1,np.pi*alpha))
    window = np.sqrt(window[:-1] / window[-1])

    if filt_len > win_len:
        pad =(filt_len - win_len) // 2
    else:
        pad = 0

    window = np.concatenate([window, window[::-1]])
    window = np.pad(window, (np.ceil(pad).astype(int), np.floor(pad).astype(int)), mode='constant')
    return torch.FloatTensor(window)[:,None]

class Spectrum_Trans_(torch.nn.Module):
    def __init__(self, spectrum_transform_type, filter_length=1024):
        """
        if MDCT:
        This module implements an MDCT using 1D convolution and 1D transpose convolutions.
        This code only implements with hop lengths that are half the filter length (50% overlap
        between frames), to ensure TDAC conditions and, as such, perfect reconstruction. 
       
        Keyword Arguments:
            filter_length {int} -- Length of filters used - only powers of 2 are supported (default: {1024})
            win_length {[type]} -- Length of the window function applied to each frame (if not specified, it
                equals the filter length). (default: {None})
        """
        super(Spectrum_Trans_, self).__init__()
        self.spectrum_transform_type = spectrum_transform_type

        self.filter_length = filter_length
        assert((filter_length % 2) == 0)

        self.hop_length    = filter_length // 2  
        self.window_length = filter_length
        self.pad_amount    = filter_length // 2

        # get window and zero center pad it to filter_length
        assert(filter_length >= self.window_length)
        self.window = kbd_window_(self.window_length, self.filter_length, alpha=4)
        self.window = self.window.cuda()
        forward_basis = mdct_basis_(filter_length).cuda()
        forward_basis *= self.window.T

        inverse_basis = forward_basis.T

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())
    
    def dct1(self, x):
        """
        Discrete cosine transform, Type I

        :param x: the input signal
        :return: the DCT-I of the signal over the last dimension
        """
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])

        return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)

    def idct1(self, X):
        """
        The inverse of DCT-I, which is just a scaled DCT-I

        Our definition if idct1 is such that idct1(dct1(x)) == x

        :param X: the input signal
        :return: the inverse DCT-I of the signal over the last dimension
        """
        n = X.shape[-1]
        return dct1(X) / (2 * (n - 1))

    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.fft.fft(v)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
        V = Vc.real * W_r - Vc.imag * W_i
        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V


    def idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

        Our definition of idct is that idct(dct(x)) == x

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
        tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
        v = torch.fft.ifft(tmp)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape).real
    
    def mdct(self, input_data):
        """Take input data (audio) to MDCT domain.
        
        Arguments:
            input_data {tensor} -- Tensor of floats, with shape (num_batch, num_samples)
        
        Returns:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)
        """
        # Pad data with win_len / 2 on either side
        input_data = input_data.unsqueeze(0)
        num_batches, num_samples = input_data.size()
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(input_data.unsqueeze(1), (np.ceil(self.pad_amount).astype(int), np.floor(self.pad_amount).astype(int),0,0), mode='constant')
        input_data = input_data.squeeze(1)

        output = F.conv1d(input_data, 
                    self.forward_basis.unsqueeze(dim=1), 
                    stride=self.hop_length, padding=0)

        # Return magnitude -> MDCT only includes real values
        return output
    
    def imdct(self, magnitude):
        """Call the inverse MDCT (iMDCT), given magnitude and phase tensors produced 
        by the ```transform``` function.
        
        Arguments:
            magnitude {tensor} -- Magnitude of MDCT with shape (num_batch, 
                num_frequencies, num_frames)
        
        Returns:
            inverse_transform {tensor} -- Reconstructed audio given magnitude and phase. Of
                shape (num_batch, num_samples)
        """
        inverse_transform = F.conv_transpose1d(magnitude, 
                            self.inverse_basis.unsqueeze(dim=1).T, 
                            stride=self.hop_length, padding=0)
        inverse_transform = (inverse_transform[..., np.ceil(self.pad_amount).astype(int):-np.floor(self.pad_amount).astype(int)]).squeeze(1)*(4/self.filter_length)

        return inverse_transform.squeeze(0)

    def fft(self, x):
        y = np.fft.fft(x.detach().cpu().numpy())
        return torch.Tensor(y)

    def ifft(self, x):
        y = np.fft.ifft(x.detach().cpu().numpy())
        return torch.Tensor(y)

    def spectrum_T(self, x):
        if self.spectrum_transform_type == 'DCT':
            y = self.dct(x.detach().cpu(), norm='ortho')
        if self.spectrum_transform_type == 'FFT':
            y = self.fft(x)
        if self.spectrum_transform_type == 'MDCT':
            y = self.mdct(x)
        return y

    def i_spectrum_T(self, x):
        if self.spectrum_transform_type == 'DCT':
            y = self.idct(x.detach().cpu(), norm='ortho')
        if self.spectrum_transform_type == 'FFT':
            y = self.ifft(x)
        if self.spectrum_transform_type == 'MDCT':
            y = self.imdct(x)
        return y
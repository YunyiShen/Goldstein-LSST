import numpy as np

def make_point_snr_cut(min_snr = 3.):
    '''
    return a function to make snr cuts to mask 
    Arg:
        min_snr: minimum SNR for a point to be included
    Return:
        cut function, taking photoband, photomag, phototime, photoerror, photosnr, photomask
            return a cutted version
    '''
    def cut(photoband, photomag, phototime, photoerror, photosnr, photomask):
        photomask[photosnr < min_snr] = 0
        return photoband * photomask, photomag * photomask, \
                phototime * photomask, photoerror * photomask, photosnr * photomask, photomask
    return cut


def make_event_snr_length_cut(maxsnr = 5, minband = 2, min_measures = 10):
    '''
    return a function to make snr cuts for event
    Arg:
        maxsnr: maximum snr to coult this band 
        minband: minmum band count to include this event
    Return:
        a function cut, taking photoband, photomag, phototime, photoerror, photosnr, photomask
            return if the event is accepted or not
    '''
    def cut(photoband, photomag, phototime, photoerror, photosnr, photomask):
        if np.sum(photomask) < min_measures:
            return False
        photosnr = photosnr * photomask # mask out non observations
        n_bands_exceeds_SNRcut = 0
        for bnd in range(6):
            idx = np.where(photoband == bnd)[0]
            if len(idx) > 0:
                n_bands_exceeds_SNRcut += int(np.sum(photosnr[idx] > maxsnr) > 0)
            if n_bands_exceeds_SNRcut >= minband:
                return True
        return False
    return cut

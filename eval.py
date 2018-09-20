import subprocess
def cal_pesq(reference, degraded, sample_rate=None, program='PESQ'):
    """ Return: PESQ score (PESQ-MOS/raw-MOS)
        Return:MOS-LQO
        reference:clean data(wav)
        degraded: recovered data(wav)
        requirement: sample_rate=8000/16000Hz, PESQ utility must be installed (862).
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')

    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')
    import subprocess
    args = [ program, '+%d' % sample_rate, reference, degraded  ]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    last_line = out.split('\n')[-2]
    if not last_line.startswith('P.862 Prediction'):
        raise ValueError(last_line)
    return tuple(map(float, last_line.split()[-2:]))

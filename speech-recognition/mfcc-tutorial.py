"""
https://ratsgo.github.io/speechbook/docs/fe/mfcc#mel-frequency-cepstral-coefficients
"""


"""
Prepare audio sample
"""
import scipy.io.wavfile

sample_rate, signal = scipy.io.wavfile.read('example.wav')

# https://ratsgo.github.io/speechbook/docs/phonetics/acoustic#digitization
print("sample_rate:",sample_rate,"KHz") # sampling count of file per 1sec
print("signal(quantization):",signal) # sequence of integers
print("length of signal:", len(signal)) 
print("length of file:", len(signal)/sample_rate)

# cut signal into 3.5sec
signal = signal[0:int(3.5 * sample_rate)]
print("len(signal):", len(signal))


"""
Preemphasis
y_t = x_t - a * x_(t-1) : first-order high-pass filter
a is preemphasis coefficient(usually 0.95 or 0.97)
1. strengthen the high frequency features, so to be even signal
2. prevent numerical problem from Fourier transform
3. improve sSignal-to-Noise Ratio(SNR)
"""
import numpy as np

pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
print("emphasized_signal", emphasized_signal)


"""
Framing
signal is non-stationary, so need to frame in a really short time(25ms)
to asume signal as stationary
"""
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
signal_length = len(emphasized_signal)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z)
print("pad_signal:",pad_signal)
print("len(pad_signal:", len(pad_signal))
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
          np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
print("indices:",indices) # 인덱스가 겹침. ex. 0~399, 160~559
print("indices.shape", indices.shape) # num_frames, frmae 당 길이(25ms * 16000 sample_rate)
frames = pad_signal[indices.astype(np.int32, copy=False)]
print("frames:",frames)
print("frames.shape:",frames.shape)


"""
Windowing
smooth boundary by applying specific function(Hamming Window) to each frames
"""
frames *= np.array([0.54 - 0.46 * np.cos((2*np.pi(n)) / (frame_length-1)) for n in range(frame_length)])


"""
Fourier Transform
"""
NFFT = 512 # 주파수 도메인으로 변환할 때 몇 개의 구간(bin)으로 분석할지 나타내는 인자(argument)
dft_frames = np.fft.rfft(frames, NFFT) # 함수의 변환 결과에서 켤레 대칭인 파트 계산을 생략


"""
Magnitude 진폭
a + b * j => sqrt(a^2 + b^2)
"""
mag_frames = np.absolute(dft_frames)


"""
Power Spectrum 파워
진폭을 구하든 파워를 구하든 복소수 형태인 이산 푸리에 변환 결과(복소수)는 모두 실수로 바뀜
"""
pow_frames = ((1.0/NFFT) * ((mag_frames) ** 2))


"""
Filter Banks
"""
nfilt = 40 # 멜스케일 필터
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1)))) # fbank[39]의 각 요소값은 해당 주파수 구간을 얼마나 살필지 가중치 역할
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right
    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

# 필터 적용
filter_bank = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)


"""
Log-Mel Spectrum
"""
filter_banks = 20 * np.log10(filter_banks) # dB


"""
MFCC
apply Inverse Fourier Transform to solve coerrelation problem
"""
# Inverse Discrete Cosine Transform
from scipy.fftpack import dct
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps + 1)] # 2-13번째 열벡터들만

print(mfcc.shape)
print(mfcc)


"""
Post Processing
"""
# Lift
(nframes, ncoeff) = mfcc.shape
cep_lifter = 22
n = np.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

# Mean Normalization
filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)


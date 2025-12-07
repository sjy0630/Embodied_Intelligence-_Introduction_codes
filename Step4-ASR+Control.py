import sherpa_onnx
from sherpa_onnx import OfflineRecognizer
from typing import Union
import librosa
import sounddevice as sd
import numpy as np
import requests

# ！！！--- 请确认你的模型路径是否正确 ---！！！
asr_path = 'model/ASR/sherpa-onnx-paraformer-zh-small-2024-03-09'
vad_path = 'model/VAD'

class ASR:
    def __init__(self):
        self._recognizer = OfflineRecognizer()
        raise NotImplementedError

    def transcribe(self, audio: Union[str, np.ndarray], sample_rate=16000) -> str:
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=sample_rate)
        s = self._recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(s)
        return s.result.text


class Whisper(ASR):
    def __init__(self, encoder_path: str, decoder_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder_path,
            decoder=decoder_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )


class Paraformer(ASR):
    def __init__(self, model_path: str, tokens_path: str, num_threads: int = 8, provider: str = 'cpu'):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            provider=provider,
        )

print('正在加载模型...')
# 请确保你的模型文件夹里确实有这些文件
try:
    asr = Paraformer(
        model_path=f'{asr_path}/model.int8.onnx',
        tokens_path=f'{asr_path}/tokens.txt',
        # provider='cuda', # 如果有显卡可以取消注释
    )
    print('模型加载完成')
except Exception as e:
    print(f"模型加载失败: {e}")
    print("请检查 asr_path 和 vad_path 变量是否指向了正确的文件夹")
    exit()

sample_rate = 16000

from sherpa_onnx import VadModelConfig, SileroVadModelConfig, VoiceActivityDetector
config = VadModelConfig(
    SileroVadModelConfig(
        model=f'{vad_path}/silero_vad.onnx',
        min_silence_duration=0.25,
    ),
    sample_rate=sample_rate
)
window_size = config.silero_vad.window_size
vad = VoiceActivityDetector(config, buffer_size_in_seconds=100)
samples_per_read = int(0.1 * sample_rate)

# ！！！--- 请确认这里的 IP 地址 ---！！！
# 你之前的视觉代码用的是 192.168.87.17，这里你贴的是 10.207.27.17
# 请根据实际情况修改！
control_url = "http://10.207.27.17:5000/control" 

def send_command(text):
    global control_url
    
    # 默认指令
    command = "STOP"
    steer = 0.0
    
    # --- 核心修改：适配新的 steer 控制逻辑 ---
    if '左' in text:
        # 左转：发送 FORWARD，并将 steer 设为 -1.0 (最大左转/原地左转)
        command = "FORWARD"
        steer = -1.0 
        print(f"识别到'左' -> 左转 (Steer: {steer})")
        
    elif '右' in text:
        # 右转：发送 FORWARD，并将 steer 设为 1.0 (最大右转/原地右转)
        command = "FORWARD"
        steer = 1.0
        print(f"识别到'右' -> 右转 (Steer: {steer})")
        
    elif '前' in text or '走' in text:
        # 前进：发送 FORWARD，steer 为 0.0 (直行)
        command = "FORWARD"
        steer = 0.0
        print(f"识别到'前' -> 前进")
        
    elif '后' in text or '退' in text:
        # 后退：发送 BACKWARD (树莓派代码里保留了对 BACKWARD 的支持)
        command = "BACKWARD"
        steer = 0.0
        print(f"识别到'后' -> 后退")
        
    elif '停' in text:
        command = "STOP"
        steer = 0.0
        print(f"识别到'停' -> 停止")
        
    else:
        # 未识别出有效指令，默认停止或忽略
        # command = "STOP"
        print(f"未识别指令，忽略")
        return # 不发送请求

    # 发送请求
    try:
        # 现在的树莓派服务器需要 'steer' 参数
        response = requests.post(control_url, json={'command': command, 'steer': steer})
        # print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"指令发送失败: {e}")

print('正在识别音频... (按 Ctrl+C 退出)')
idx = 1
buffer = []
try:
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)  # a blocking read
            samples = samples.reshape(-1)

            buffer = np.concatenate([buffer, samples])
            while len(buffer) > window_size:
                vad.accept_waveform(buffer[:window_size])
                buffer = buffer[window_size:]

            while not vad.empty():
                text = asr.transcribe(vad.front.samples, sample_rate=sample_rate)

                vad.pop()
                if len(text):
                    print(f'第{idx}句：{text}')
                    send_command(text)
                    idx += 1
except KeyboardInterrupt:
    sd.stop()
    print('\n识别结束')